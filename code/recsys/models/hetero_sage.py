class HeteroGraphSAGE(nn.Module):
  """
  Encoder GraphSAGE heterogêneo.

  Todos os tipos de nó são projetados para `hidden_channels` na entrada
  e permanecem nessa dimensão em todas as camadas, de modo que
  `out[a] @ out[b].T` é uma pontuação válida para qualquer par (a, b).

  Parâmetros
  ----------
  metadata : tuple
      `(node_types, edge_types)` — pode vir de `hetero_data.metadata()`.
  in_channels_dict : dict[str, int]
      Dimensão de entrada por tipo de nó. Use `-1` se o nó não tiver
      features e você quiser usar um embedding aprendível.
  hidden_channels : int
      Dimensão de saída (mesma para todos os nós).
  num_layers : int
      Número de camadas de message passing.
  dropout : float
  aggr : str
      Agregação entre tipos de aresta no `HeteroConv` ('mean' é seguro).
  normalize : bool
      Se True, aplica L2-norm na saída, scores em [-1, 1].
  normalize_intern: bool
      Se True, aplica L2-norm nas camadas internas, scores em [-1, 1].
  """
  def __init__(
    self,
    metadata,
    in_channels_dict: dict,
    hidden_channels: int,
    num_layers: int,
    dropout: float,
    aggr: str,
    normalize: bool,
    input_proj: bool,
  ):
    super().__init__()
    self.node_types, self.edge_types = metadata
    self.hidden_channels = hidden_channels
    self.num_layers = num_layers
    self.dropout = dropout
    self.normalize = normalize

    if input_proj:
      self.input_proj = nn.ModuleDict()
      for nt in self.node_types:
        in_ch = in_channels_dict[nt]
        if in_ch == -1:
          raise ValueError(
            f"Defina o número de nós para '{nt}' via add_module; "
            "ou use features."
          )
        self.input_proj[nt] = Linear(in_ch, hidden_channels)

    self.convs = nn.ModuleList()
    self.norms = nn.ModuleList()

    for i in range(num_layers):
      # Decide as dimensões de entrada por aresta.
      # - input_proj=True: tudo já está em hidden_channels (projeção feita antes).
      # - input_proj=False, i==0: dims brutas por nó.
      # - input_proj=False, i>0: hidden_channels (assumindo que todo nó é destino
      #   de ao menos uma aresta na camada 0; caso contrário, use input_proj=True).
      if input_proj or i > 0:
        conv_dict = {
          et: SAGEConv(
              (hidden_channels, hidden_channels),
              hidden_channels,
              aggr="mean",
          )
          for et in self.edge_types
        }
      else:  # i == 0 and not input_proj
        conv_dict = {
          et: SAGEConv(
              (in_channels_dict[et[0]], in_channels_dict[et[2]]),
              hidden_channels,
              aggr="mean",
          )
          for et in self.edge_types
        }
      self.convs.append(HeteroConv(conv_dict, aggr=aggr))
      self.norms.append(
          nn.ModuleDict({nt: nn.LayerNorm(hidden_channels) for nt in self.node_types})
      )

def forward(self, x_dict, edge_index_dict):
  if getattr(self, "input_proj", None) is not None:
    h = {nt: self.input_proj[nt](x) for nt, x in x_dict.items()}
  else:
    h = dict(x_dict)  # dims brutas

  h = {nt: F.dropout(x, p=self.dropout, training=self.training)
    for nt, x in h.items()}

  for conv, norm in zip(self.convs, self.norms):
    h_new = conv(h, edge_index_dict)

    for nt in self.node_types:
      if nt not in h_new:
        h_new[nt] = h[nt]
      h_new[nt] = norm[nt](h_new[nt])
      h_new[nt] = F.relu(h_new[nt])
      h_new[nt] = F.dropout(h_new[nt], p=self.dropout, training=self.training)
    h = h_new

  if self.normalize:
    h = {nt: F.normalize(x, p=2, dim=-1) for nt, x in h.items()}
  return h
