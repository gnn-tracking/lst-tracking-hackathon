from gnn_tracking.utils.loading import TrackingDataModule

default_data_module = TrackingDataModule(
    train=dict(
        dirs=[
            "/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/lst_data_v1/processed/"
        ],
        stop=150,
        # sample_size=800,
    ),
    val=dict(
        dirs=[
            "/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/lst_data_v1/processed/"
        ],
        start=150,
        stop=155,
    ),
    test=dict(
        dirs=[
            "/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/lst_data_v1/processed/"
        ],
        start=170,
        stop=175,
    ),
    cpus=3,
    identifier="lst_v1",
    # could also configure a 'test' set here
)
