import torch


class EnsembleAAESpecialized(torch.nn.Module):
    def __init__(self, general_model, specialized_model):
        super(EnsembleAAESpecialized, self).__init__()
        self.general_model = general_model
        self.specialized_model = specialized_model

    def forward(self, input_ids_sae, attention_mask_sae, token_type_ids_sae, input_ids_aae, attention_mask_aae,
                token_type_ids_aae, is_aae_06):
        general_output = self.general_model(input_ids=input_ids_sae,
                                       attention_mask=attention_mask_sae,
                                       token_type_ids=token_type_ids_sae,
                                       )

        specialized_output = self.specialized_model(input_ids=input_ids_aae,
                                           attention_mask=attention_mask_aae,
                                           token_type_ids=token_type_ids_aae)

        general_logits = general_output['logits']
        specialized_logits = specialized_output['logits']

        assert general_logits.shape == specialized_logits.shape

        # this is precomputed but the Blodgett Model can also be included here instead
        is_aae_mask = (torch.argmax(general_logits, 1)) & (is_aae_06.type(torch.LongTensor).to('cuda'))
        is_aae_mask = is_aae_mask.unsqueeze(1).repeat(1, 2).type(torch.BoolTensor).to('cuda')

        # Applies the specialized learner prediction when prediction of general is 1 and the dialect model
        # estimates sample to be aae,
        ensemble_logits = torch.where(is_aae_mask, specialized_logits, general_logits)

        return ensemble_logits
