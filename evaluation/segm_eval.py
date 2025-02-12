import os
from typing import TYPE_CHECKING
import cv2
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metric_utils import get_metric_name

from evaluation.metric import PixelMetric

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate
    from benchmark.naive_segmentation import SegmentationTemplate

class SegmMetrics(PluginMetric[dict]):
    """
    Metric used to compute the detection and segmentation metrics using the
    dataset-specific API.

    Metrics are returned after each evaluation experience.

    This metric can also be used to serialize model outputs to JSON files,
    by producing one file for each evaluation experience. This can be useful
    if outputs have to been processed later (like in a competition).

    If no dataset-specific API is used, the COCO API (pycocotools) will be used.
    """

    def __init__(
        self,
        *,
        save_folder=None,
        filename_prefix="model_output",
    ):
        """
        Creates an instance of DetectionMetrics.

        :param save_folder: path to the folder where to write model output
            files. Defaults to None, which means that the model output of
            test instances will not be stored.
        :param filename_prefix: prefix common to all model outputs files.
            Ignored if `save_folder` is None. Defaults to "model_output"
        """
        super().__init__()

        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)

        self.save_folder = save_folder
        """
        The folder to use when storing the model outputs.
        """

        self.filename_prefix = filename_prefix
        """
        The file name prefix to use when storing the model outputs.
        """

        self.evaluator = None
        """
        Main evaluator object to compute metrics.
        """

        self.save = save_folder is not None
        """
        If True, model outputs will be written to file.
        """

    def reset(self) -> None:
        self.evaluator = None

    def update(self, res):
        out_dir = '/{path_to_files}/'

        y = res['outputs']
        cls_gt = res['targets']

        cls = (y > 0.5).cpu()
        cls = cls.numpy()

        cls_gt = cls_gt.cpu().numpy()
        y_true = cls_gt.ravel()
        y_pred = cls.ravel()

        for im_id, pred in zip(res['names'], cls):
            cv2.imwrite(os.path.join(out_dir, im_id + '.png'), pred.squeeze() * 255)

        self.evaluator.forward(y_true, y_pred)

    def result(self):
        # result_dict may be None if not running in the main process
        result_dict = self.evaluator.summary_all()

        return result_dict


    def before_eval_exp(self, strategy) -> None:
        super().before_eval_exp(strategy)

        self.reset()
        self.evaluator = PixelMetric(2)

    def after_eval_iteration(self, strategy: "SegmentationTemplate") -> None:
        super().after_eval_iteration(strategy)
        self.update(strategy.detection_predictions)

    def after_eval_exp(self, strategy: "SupervisedTemplate"):
        super().after_eval_exp(strategy)

        packaged_results = self._package_result(strategy)
        return packaged_results

    def _package_result(self, strategy):
        base_metric_name = get_metric_name(
            self, strategy, add_experience=True, add_task=False
        )
        plot_x_position = strategy.clock.train_iterations
        result_dict = self.result()

        if result_dict is None:
            return

        metric_values = []
        for metric_key, metric_value in result_dict.items():
            metric_name = base_metric_name + f"/{metric_key}"
            metric_values.append(
                MetricValue(
                    self, metric_name, metric_value, plot_x_position
                )
            )

        return metric_values

    def __str__(self):
        return "SegmMetrics"


def make_segm_metrics(
    save_folder=None,
    filename_prefix="model_output",
):
    """
    Returns an instance of :class:`DetectionMetrics` initialized for the LVIS
    dataset.

    :param save_folder: path to the folder where to write model output
            files. Defaults to None, which means that the model output of
            test instances will not be stored.
    :param filename_prefix: prefix common to all model outputs files.
        Ignored if `save_folder` is None. Defaults to "model_output"
    """
    return SegmMetrics(
        save_folder=save_folder,
        filename_prefix=filename_prefix,
    )

