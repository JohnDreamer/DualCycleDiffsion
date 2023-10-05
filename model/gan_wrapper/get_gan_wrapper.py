

def get_gan_wrapper(args, target=False):

    kwargs = {}
    for kw, arg in args:
        if kw != 'gan_type':
            if (not kw.startswith('source_')) and (not kw.startswith('target_')):
                kwargs[kw] = arg
            else:
                if target and kw.startswith('target_'):
                    final = kw[len('target_'):]
                    kwargs[f'source_{final}'] = arg
                elif (not target) and kw.startswith('source_'):
                    kwargs[kw] = arg

    if args.gan_type == "DualCycleStochasticTextOpt":
        from .dual_cycle_stochastic_text_wrapper_sd import DualCycleDiffusion
        return DualCycleDiffusion(**kwargs)
    elif args.gan_type == "DualCycleLatentStochasticTextOpt":
        from .dual_cycle_stochastic_text_wrapper_ldm import DualCycleDiffusion
        return DualCycleDiffusion(**kwargs)
    else:
        raise ValueError()

