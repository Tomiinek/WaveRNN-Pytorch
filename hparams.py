import tensorflow as tf

# Default hyperparameters:
hparams = tf.contrib.training.HParams(
    name="WaveRNN",
    num_workers=8,
    # Input type:
    # 1. raw [-1, 1]
    # 2. mixture [-1, 1]
    # 3. bits [0, 512]
    # 4. mulaw[0, mulaw_quantize_channels]
    #
    input_type='bits',
    #
    # distribution type, currently supports only 'beta' and 'mixture'
    distribution='beta',  # or "mixture"
    log_scale_min=-32.23619130191664,  # = float(np.log(1e-7))
    quantize_channels=65536,  # quantize channel used for compute loss for mixture of logistics
    #
    # for Fatcord's original 9 bit audio, specify the audio bit rate. Note this corresponds to network output
    # of size 2**bits, so 9 bits would be 512 output, etc.
    bits=10,
    # for mu-law
    mulaw_quantize_channels=512,
    # note: r9r9's deepvoice3 preprocessing is used instead of Fatchord's original.
    # --------------
    # audio processing parameters
    num_mels=80,
    fmin=95,
    fmax=7600,
    n_fft=2048,
    hop_size=200,
    win_size=800,
    sample_rate=16000,

    min_level_db=-100,
    ref_level_db=20,
    rescaling=False,
    rescaling_max=0.999,

    #Mel and Linear spectrograms normalization/scaling and clipping
    signal_normalization = True, #Whether to normalize mel spectrograms to some predefined range (following below parameters)
    allow_clipping_in_normalization = True, #Only relevant if mel_normalization = True
    symmetric_mels = True, #Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence)
    max_abs_value = 4., #max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not be too big to avoid gradient explosion,

    #Contribution by @begeekmyfriend
    #Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude levels. Also allows for better G&L phase reconstruction)
    preemphasize = False, #whether to apply filter
    preemphasis = 0.97, #filter coefficient.

    magnitude_power=2., #The power of the spectrogram magnitude (1. for energy, 2. for power)

	# Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
	# It's preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
	# Does not work if n_ffit is not multiple of hop_size!!
	use_lws=False, #Only used to set as True if using WaveNet, no difference in performance is observed in either cases.
	silence_threshold=2, #silence threshold used for sound trimming for wavenet preprocessing


    # ----------------
    #
    # ----------------
    # model parameters
    rnn_dims=256,
    fc_dims=128,
    pad=2,
    # note upsample factors must multiply out to be equal to hop_size, so adjust
    # if necessary (i.e 4 x 5 x 10 = 200)
    upsample_factors=(4, 5, 10),
    compute_dims=64,
    res_out_dims=32*2, #aux output is fed into 2 downstream nets
    res_blocks=3,
    # ----------------
    #
    # ----------------
    # training parameters
    batch_size=64,
    nepochs=5000,
    save_every_step=10000,
    evaluate_every_step=10000,
    # seq_len_factor can be adjusted to increase training sequence length (will increase GPU usage)
    seq_len_factor=7,

    grad_norm=10,
    # learning rate parameters
    initial_learning_rate=1e-3,
    lr_schedule_type='noam',  # or 'noam'

    # for step learning rate schedule
    step_gamma=0.5,
    lr_step_interval=15000,

    # sparsification
    start_prune=80000,
    prune_steps=80000,  # 20000
    sparsity_target=0.90,
    sparsity_target_rnn=0.90,
    sparse_group=4,

    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1e-8,
    amsgrad=False,
    weight_decay=0, #1e-5,
    fix_learning_rate=None,
    # modify if one wants to use a fixed learning rate, else set to None to use noam learning rate
    # -----------------
    batch_size_gen=32,
)

hparams.seq_len = hparams.seq_len_factor * hparams.hop_size

# for noam learning rate schedule
hparams.noam_warm_up_steps = 2000 * (hparams.batch_size // 16)
