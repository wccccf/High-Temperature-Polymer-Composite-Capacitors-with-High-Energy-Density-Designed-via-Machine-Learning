import torch


def compute_mean_mad(dataset, properties, dataset_name):
        return compute_mean_mad_from_dataloader(dataset, properties)

def compute_mean_mad_from_dataloader(dataset, properties):
    property_norms = {}
    for property_key in properties:
        values = dataset.data[property_key]
        mean = torch.mean(values)
        ma = torch.abs(values - mean)
        mad = torch.mean(ma)
        max_value = torch.max(values)
        min_value = torch.min(values)
        property_norms[property_key] = {}
        property_norms[property_key]['mean'] = mean
        property_norms[property_key]['mad'] = mad
        property_norms[property_key]['max'] = max_value
        property_norms[property_key]['min'] = min_value
    return property_norms


def prepare_context(conditioning, minibatch, property_norms):
    # batch_size = minibatch['batch'][-1] + 1
    context_node_nf = 0
    context_list = []
    for key in conditioning:
        properties = minibatch[key]
        properties = (properties+5.0)/3
        if len(properties.size()) == 1:
            # Global feature.
            # assert properties.size() == (batch_size,)
            properties = properties.index_select(0, minibatch['batch'])
            context_list.append(properties.unsqueeze(1))
            context_node_nf += 1
        elif len(properties.size()) == 2 or len(properties.size()) == 3:
            # Node feature.
            # assert properties.size(0) == batch_size

            context_key = properties

            # Inflate if necessary.
            if len(properties.size()) == 2:
                context_key = context_key.unsqueeze(2)

            context_list.append(context_key)
            context_node_nf += context_key.size(2)
        else:
            raise ValueError('Invalid tensor size, more than 3 axes.')
    # Concatenate
    context = torch.cat(context_list, dim=1)
    # Mask disabled nodes!
    assert context.size(1) == context_node_nf
    return context