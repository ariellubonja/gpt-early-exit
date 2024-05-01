from matplotlib import pyplot as plt

def plot_intermediate_model_outputs(model, keys = ['initial_hidden_states', 'post_ln1_hidden_states', 'attn_projection_output', 'post_attn_residual_hidden_states', 'post_cross_attn_hidden_states', 'post_ln2_hidden_states', 'post_feed_fwd_hidden_states', 'post_feed_fwd_residual_hidden_states']):
    for i in range(len(model.h)):
        int_out = model.h[i].intermediate_outputs
        for key in keys:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            im = ax1.imshow(int_out[key][0])#, aspect='auto', interpolation='nearest')
            ax1.set_title(f'Layer {i} {key} - Image')
            ax1.figure.colorbar(im, ax=ax1)
            ax2.hist(int_out[key][0])#.ravel())#, bins=20, color='blue')
            ax2.set_title(f'Layer {i} {key} - Histogram')
            plt.show()