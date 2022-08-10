import torch

def position_coding(inputs, nbins, min_input=-1, max_input=1):
    """
    Linear population coding where the inputs are put in nbins per input feature
    """

    if not torch.is_tensor(inputs):
        inputs = torch.as_tensor(inputs.to_numpy()).unsqueeze(0)

    batch_size, seq_size, nfeatures = inputs.size()
    spikes = torch.zeros([batch_size, seq_size, nfeatures * nbins], device=inputs.device)
    
    for i in range(nfeatures):
        data = inputs[:, :, i]

        bin_size = (max_input - min_input) / nbins
        neuron_vals = torch.linspace(min_input + bin_size / 2, max_input - bin_size / 2, nbins, device=inputs.device)
        for j in range(batch_size):
            for k in range(seq_size):
                diff = neuron_vals - data[j, k]
                distance = torch.square(diff)
                min_index = torch.argmin(distance)
                spikes[j, k, i * nbins + min_index] = 1
    return spikes

def grf_coding(inputs, nbins, min_input=-1, max_input=1):

    if not torch.is_tensor(inputs):
        inputs = torch.as_tensor(inputs.to_numpy()).unsqueeze(0)

    batch_size, seq_size, nfeatures = inputs.size()
    spikes = torch.zeros([batch_size, seq_size, nfeatures * nbins], device=inputs.device)
    

    sigma = (max_input - min_input)/(nbins * 2 * 0.5)
    
    for i in range(nfeatures):
        data = inputs[:, :, i]

        bin_size = (max_input - min_input) / nbins
        neuron_vals = torch.linspace(min_input + bin_size / 2, max_input - bin_size / 2, nbins, device=inputs.device)
        neuron_probs = torch.distributions.Normal(neuron_vals, sigma)

        for j in range(batch_size):
            for k in range(seq_size):
                neuron_outputs_cdf = neuron_probs.cdf(data[j, k])
                stacked = torch.stack([neuron_outputs_cdf, 1-neuron_outputs_cdf])
                neuron_outputs = torch.min(stacked, axis=0).values
                spikes[j, k, i * nbins : i * nbins + nbins] = (neuron_outputs - torch.rand(size=neuron_outputs.size(), device=inputs.device)).gt(0.0).float()
    return spikes


def stochastic_position_coding(inputs, nbins, min_input=-1, max_input=1):

    if not torch.is_tensor(inputs):
        inputs = torch.as_tensor(inputs.to_numpy()).unsqueeze(0)

    batch_size, seq_size, nfeatures = inputs.size()
    spikes = torch.zeros([batch_size, seq_size, nfeatures * nbins], device=inputs.device)

    sigma = (max_input - min_input)/(nbins + 2)
    
    for i in range(nfeatures):
        data = inputs[:, :, i]

        bin_size = (max_input - min_input) / nbins
        neuron_vals = torch.linspace(min_input + bin_size / 2, max_input - bin_size / 2, nbins, device=inputs.device)
        neuron_probs = torch.distributions.Normal(neuron_vals, sigma)

        for j in range(batch_size):
            for k in range(seq_size):
                neuron_outputs = neuron_probs.log_prob(data[j, k]).exp()
                neuron_outputs = neuron_outputs / torch.linalg.norm(neuron_outputs)
                neuron_choice = torch.multinomial(neuron_outputs, 1)
                spikes[j, k, i * nbins + neuron_choice] = 1
    return spikes


def soft_position_coding(inputs, nbins, min_input=-1, max_input=1):
    """
    Linear population coding where the inputs are put in nbins per input feature, input is based on gaussian response
    """

    if not torch.is_tensor(inputs):
            inputs = torch.as_tensor(inputs.to_numpy()).unsqueeze(0)

    batch_size, seq_size, nfeatures = inputs.size()
    currents = torch.zeros([batch_size, seq_size, nfeatures * nbins], device=inputs.device)
    sigma = (max_input - min_input)/(nbins + 2) # this is a tuning parameters

    for i in range(nfeatures):
        data = inputs[:, :, i]

        bin_size = (max_input - min_input) / nbins
        neuron_vals = torch.linspace(min_input + bin_size / 2, max_input - bin_size / 2, nbins, device=inputs.device)
        neuron_probs = torch.distributions.Normal(neuron_vals, sigma)

        for j in range(batch_size):
            for k in range(seq_size):
                neuron_outputs = neuron_probs.log_prob(data[j, k]).exp()
                neuron_outputs = neuron_outputs / torch.linalg.norm(neuron_outputs)
                currents[j, k, i * nbins : i * nbins + nbins] = neuron_outputs
    return currents