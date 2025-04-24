from matplotlib import pyplot as plt
import numpy as np


def plot_waveform(waveform: np.ndarray, sample_rate: int, plot_width=5, plot_height=2, fig_save_path=None, show=True):
    """
    Plot the waveform and spectrogram of the audio signal.
    Args:
        waveform (np.ndarray): The audio signal to plot.
        sample_rate (int): The sample rate of the audio signal.
        fig_save_path (str, optional): Path to save the figure. If None, the figure will not be saved.
        show (bool, optional): Whether to show the figure. Defaults to True.
    """
    num_channels, num_frames = waveform.shape
    time_axis = np.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels * 2, 1, figsize=(plot_width, plot_height * num_channels))
    wav_axes = axes[:num_channels]
    wavspec_axes = axes[num_channels:]
    # spec_axes = axes[num_channels * 2 :]

    for c in range(num_channels):
        wav_axes[c].plot(time_axis, waveform[c], linewidth=1)
        wav_axes[c].set_xlim([0, time_axis[-1]])
        wav_axes[c].grid(True)

        cax = wavspec_axes[c].specgram(waveform[c] + 1e-10, Fs=sample_rate)
        # figure.colorbar(cax[-1], ax=wavspec_axes[c])
        # im = spec_axes[c].imshow(librosa.power_to_db(specgram[c]), origin="lower", aspect="auto")
        # figure.colorbar(im, ax=spec_axes[c], format="%+2.0f dB")
        # if num_channels > 1:
        #     wav_axes[c].set_ylabel(f"#{c + 1}")
        #     wavspec_axes[c].set_ylabel(f"#{c + 1}")
        #     spec_axes[c].set_ylabel(f"#{c + 1}")
        # if c == 0:
        #     wav_axes[c].set_title("Original Waveform")
        #     wavspec_axes[c].set_title("Original Spectrogram")
        #     spec_axes[c].set_title("Featured Spectrogram")
    figure.tight_layout()
    if fig_save_path:
        figure.savefig(fig_save_path)
    if show:
        plt.show(block=False)
    plt.close(figure)


if __name__ == "__main__":
    # Example usage
    import torchaudio
    import sys
    import os
    input = sys.argv[1] if len(sys.argv) > 1 else "example.wav"
    if not os.path.exists(input):
        print(f"Audio file {input} does not exist.")
        sys.exit(1)
    print("Available backends of torchaudio:", torchaudio.list_audio_backends())
    waveform, sample_rate = torchaudio.load(input)
    print(f"Waveform shape: {waveform.shape}")
    print(f"Sample rate: {sample_rate}")
    waveform_save_path = os.path.splitext(input)[0] + ".png"
    plot_waveform(waveform.numpy(), sample_rate, 8, 3,  waveform_save_path, show=False)
    del waveform
    print("Waveform plot saved to: ", waveform_save_path)