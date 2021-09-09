def set_device(use_gpu, multi_gpu, _log):
    # Decide which device to use.
    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError('use_gpu is True but CUDA is not available')

    if use_gpu:
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')

    if multi_gpu and torch.cuda.device_count() == 1:
        raise RuntimeError('Multiple GPU training requested, but only one GPU is available.')

    if multi_gpu:
        _log.info('Using all {} GPUs available'.format(torch.cuda.device_count()))

    return device 