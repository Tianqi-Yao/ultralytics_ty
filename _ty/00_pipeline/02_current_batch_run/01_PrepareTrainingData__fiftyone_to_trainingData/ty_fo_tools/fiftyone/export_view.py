# ty_fo_tools/fo_export.py
from __future__ import annotations

from pathlib import Path
from typing import Type, Union

import fiftyone.types as fot
import fiftyone.core.view as fov


def export_view_to_coco(
    *,
    view: fov.DatasetView,
    export_dir: Union[str, Path],
    label_field: str,
    dataset_type: Type[fot.Dataset] = fot.COCODetectionDataset,
    export_media: bool = True,
    data_dirname: str = "01_data",
    labels_filename: str = "01_all_labels.json",
) -> Path:
    """
    Export a FiftyOne view to COCO format under `export_dir`.

    It will create:
        - `<export_dir>/<data_dirname>/` for images (if export_media=True)
        - `<export_dir>/<labels_filename>` for annotations

    Parameters
    ----------
    view
        FiftyOne DatasetView (e.g., session.view or dataset.view()).
    export_dir
        Output directory.
    label_field
        Label field name in the view.
    dataset_type
        e.g. fot.COCODetectionDataset or fot.COCOInstancesDataset.
    export_media
        Whether to export media files.
    data_dirname
        Subfolder name for exported media.
    labels_filename
        File name for exported labels.

    Returns
    -------
    Path
        The export directory path.
    """
    export_dir = Path(export_dir).expanduser().resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    data_path = export_dir / data_dirname
    labels_path = export_dir / labels_filename

    view.export(
        export_dir=str(export_dir),
        dataset_type=dataset_type,
        label_field=label_field,
        export_media=export_media,
        data_path=str(data_path),
        labels_path=str(labels_path),
    )

    return export_dir
