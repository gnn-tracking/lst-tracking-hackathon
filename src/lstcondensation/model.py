from gnn_tracking.models.edge_classifier import ECFromChkpt
from gnn_tracking.models.resin import ResIN
from gnn_tracking.models.track_condensation_networks import ModularGraphTCN
from pytorch_lightning.core.mixins import HyperparametersMixin
from torch import Tensor, nn
from torch_geometric.data import Data


class LSGraphTCN(nn.Module, HyperparametersMixin):
    def __init__(
        self,
        *,
        node_indim: int,
        edge_indim: int,
        h_dim=5,
        e_dim=4,
        h_outdim=2,
        hidden_dim=40,
        L_hc=3,
        alpha_hc: float = 0.5,
        ec: ECFromChkpt | None = None,
        ec_thld: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        hc_in = ResIN(
            node_dim=h_dim,
            edge_dim=e_dim,
            object_hidden_dim=hidden_dim,
            relational_hidden_dim=hidden_dim,
            alpha=alpha_hc,
            n_layers=L_hc,
        )
        self._gtcn = ModularGraphTCN(
            hc_in=hc_in,
            node_indim=node_indim,
            edge_indim=edge_indim,
            h_dim=h_dim,
            e_dim=e_dim,
            h_outdim=h_outdim,
            hidden_dim=hidden_dim,
            ec=ec,
            ec_threshold=ec_thld,
            feed_edge_weights=True,
            mask_orphan_nodes=True,
        )

    def forward(
        self,
        data: Data,
    ) -> dict[str, Tensor]:
        return self._gtcn.forward(data=data)
