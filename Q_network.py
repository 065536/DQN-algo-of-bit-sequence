import torch
import torch.nn as nn
import torch.optim as optim

class TransformerQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TransformerQNetwork, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=hidden_dim
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, generated_sequence, target_sequence):

        generated_embedding = self.embedding(generated_sequence)
        target_embedding = self.embedding(target_sequence)
        combined_embedding = torch.cat([generated_embedding, target_embedding], dim=2)

        combined_embedding = combined_embedding.permute(2, 0, 1, 3).contiguous()
        combined_embedding = combined_embedding.view(combined_embedding.size(0), combined_embedding.size(1), -1)

        transformer_output = self.transformer(combined_embedding, combined_embedding)
        transformer_output = transformer_output.mean(dim = 0)

        output = self.fc(transformer_output)
        return output.squeeze(0)
    