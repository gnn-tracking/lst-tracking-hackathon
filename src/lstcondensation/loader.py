from gnn_tracking.utils.loading import TrackingDataModule

default_data_module = TrackingDataModule(
    train=dict(
        dirs=[
            "/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/lst_data_v0/processed/"
        ],
        stop=150,
        # sample_size=800,
    ),
    val=dict(
        dirs=[
            "/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/lst_data_v0/processed/"
        ],
        start=150,
        stop=155,
    ),
    cpus=3,
    # could also configure a 'test' set here
)
