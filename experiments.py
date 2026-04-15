from copy import deepcopy

from ray import tune

from loader.preselected_labels import preselected_labels


def decoder_variant(cfg, dec, crop):
    cfg['model']['replace_stride_with_dilation'] = [False, False, True]
    if dec in [5, 6]:
        cfg['model']['depth_args'] = {
            'intermediate_aspp': True,
            'aspp_rates': [6, 12, 18],
            'num_ch_dec': [64, 128, 128, 256, 256],
            'max_scale_size': crop
        }
        load_backbone = (dec in [6])
    elif dec == 9:
        cfg['model']['depth_args'] = {
            'intermediate_aspp': True,
            'aspp_rates': [6, 12, 18],
            'num_ch_dec': [64, 64, 128, 128, 256],
            'batch_norm': True,
            'max_scale_size': crop
        }
        load_backbone = False
    else:
        raise NotImplementedError

    return cfg, load_backbone

def setup_optimizer(cfg, opt, lr, blr, plr, slr, gclip):
    cfg["training"]["optimizer"] = {
        "name": opt,
        "lr": lr,
        "backbone_lr": blr,
    }
    if plr is not None:
        cfg["training"]["optimizer"]["pose_lr"] = plr
    if slr is not None:
        cfg["training"]["optimizer"]["segmentation_lr"] = slr
    if opt == "sgd":
        cfg["training"]["optimizer"].update({
            "momentum": 0.9,
            "weight_decay": 0.0005
        })
    cfg["training"]["clip_grad_norm"] = gclip
    return cfg


def lr_schedule(cfg, lr_sch, max_iter, step=30e3):
    if lr_sch == "step":
        cfg["training"]["lr_schedule"] = {
            "name": "step_lr", "step_size": int(50e3), "gamma": 0.1
        }
    elif lr_sch == "step2":
        cfg["training"]["lr_schedule"] = {
            "name": "multi_step", "milestones": [int(30e3), int(40e3), int(50e3)], "gamma": 0.5
        }
    elif lr_sch == "step30":
        cfg["training"]["lr_schedule"] = {
            "name": "step_lr", "step_size": int(30e3), "gamma": 0.1
        }
    elif lr_sch == "stepx":
        cfg["training"]["lr_schedule"] = {
            "name": "step_lr", "step_size": int(step), "gamma": 0.1
        }
    elif lr_sch == "poly":
        cfg['training']['lr_schedule'] = {
            'name': 'poly_lr_2', 'power': 0.9, 'max_iter': max_iter
        }
    else:
        raise NotImplementedError

    return cfg

def setup_dataset(cfg, dataset, crop, lr_sch):
    train_iters = {"cityscapes": int(40e3), "mapillary": int(40e3), "camvid": int(20e3)}[dataset]
    step = {"cityscapes": int(30e3), "mapillary": int(30e3), "camvid": int(15e3)}[dataset]
    final_val_interval = {"cityscapes": 500, "mapillary": 1000, "camvid": 500}[dataset]
    w, h = {"cityscapes": (1024, 512), "mapillary": (704, 512), "camvid": (672, 512)}[dataset]
    cfg['data'].update({
        'dataset': dataset,
        'path': {"cityscapes": "MachineConfig.CITYSCAPES_DIR",
                 "camvid": "MachineConfig.CAMVID_DIR",
                 "mapillary": "MachineConfig.MAPILLARY_DIR"}[dataset],
        'val_split': {"cityscapes": "val", "mapillary": "validation", "camvid": "test"}[dataset],
    })
    cfg['monodepth_options']['height'] = h
    cfg['monodepth_options']['width'] = w
    cfg['monodepth_options']['crop_h'] = crop[0]
    cfg['monodepth_options']['crop_w'] = crop[1]
    cfg['training']['train_iters'] = train_iters
    cfg = lr_schedule(cfg, lr_sch, train_iters, step=step)
    cfg['training']['val_interval'][str(int(step))] = final_val_interval

    return cfg

def set_segmentation_args(cfg, seg_init, layers, head_inter, output_stride, head_dropout=0.1):
    cfg['model']['segmentation_args'] = {
        'weights': seg_init,
        'layers': layers,
        'head_inter_channels': 64,
        'layer_out_channels': 64,
        'head_dropout': head_dropout,
        'layer_dropout': 0,
        'head_inter': head_inter,
        'output_stride': output_stride
    }
    return cfg

def subsets(dataset):
    if dataset == "cityscapes":
        return [
            100,
            372,
            744,
            # 2975
        ]
    elif dataset == "camvid":
        return [
            # 50,
            100,
            # 367
        ]
    elif dataset == "mapillary":
        return [
            # 100,
            2250,
            # 18000
        ]
    else:
        raise NotImplementedError(dataset)

def generate_experiment_cfgs(base_cfg, id):
    cfgs = []
    # Main Semi-Supervised Experiments (only Pretraining)
    if id == 210:
        layers = [9]
        output_stride = 1
        head_inter = False
        opt = "sgd"
        lr = 1e-2
        blr = 1e-3
        gclip = 10
        dataset = "cityscapes" # available: cityscapes, camvid, mapillary
        lr_sch = "stepx"
        for dec, dec_params, crop, batch_size in [
            (6, "lr5_fd2_crop512x512bs4", (512, 512), 2),
            # (6, "lr5_fd0_crop512x512bs4", (512, 512), 2), # for pretraining w/o feature distance loss
        ]:
            for seed in [
                7,
                25,
                42
            ]:
                mono_pretrain = f'mono_cityscapes_1024x512_r101dil_aspp_dec{dec}_{dec_params}'
                for n_subset in subsets(dataset):
                    dc_ft = 0
                    dc_m = 0.03
                    pres_method = "ds_us"  # available: "ent", "ds", "us", "ds_us"
                    for name, seg_init, teacher_init, ema, mix_mask, only_unlabeled, mix_use_gt, preselect, mix_video in [
                        ('scratch', 'none', 'none', False, None, True, False, False, False),
                        # (f'sel_{pres_method}_scratch', 'none', 'none', False, None, True, False, True, False),
                        # ('scratch_ema', 'none', 'none', True, None, True, False, False, False),
                        # ('scratch_classmix', 'none', 'none', True, "class", True, False, False, False),
                        # ('scratch_classmix_video', 'none', 'none', True, "class", False, False, False, True),
                        # ("scratch_classmixgt", 'none', 'none', True, "class", False, True, False, False),
                        # ("scratch_depthmixgt", 'none', 'none', True, "depthcomp", False, True, False, False),
                        # ('transfer', mono_pretrain, mono_pretrain, False, None, True, False, False, False),
                        # ('transfer_ema', mono_pretrain, mono_pretrain, True, None, True, False, False, False),
                        # ('transfer_classmix', mono_pretrain, mono_pretrain, True, "class", True, False, False, False),
                        # ('transfer_classmixgtall', mono_pretrain, mono_pretrain, True, "class", False, True, False, False),
                        # (f'transfer_dcompgt{dc_m}{dc_ft}', mono_pretrain, mono_pretrain, True, "depthcomp", False, True,
                        #  False, False),
                        # (f'sel_{pres_method}_transfer_dcompgt{dc_m}{dc_ft}', mono_pretrain, mono_pretrain, True, "depthcomp", False,
                        #  True, True, False),
                    ]:
                        name = name.replace('.', '').replace(' ', '').replace(',', 'i').replace('(', 'I').replace(')',
                                                                                                                  'I')
                        restrict_mode = "fixed" if preselect else "random"
                        unlab_cfg = {
                            "consistency_weight": 1.0,
                            "mix_mask": mix_mask,
                            "color_jitter": True,
                            "blur": True,
                            "only_unlabeled": only_unlabeled,
                            "only_labeled": False,
                            "mix_video": mix_video,
                            "mix_use_gt": mix_use_gt,
                            "depthcomp_margin": dc_m,
                            "depthcomp_foreground_threshold": dc_ft,
                            "backward_first_pseudo_label": False,
                            "debug_image": True
                        } if ema else None
                        unlab_str = "" if not ema else f"_Unlab{unlab_cfg['consistency_weight']}{unlab_cfg['mix_mask']}" + \
                                                       ("jit" if unlab_cfg["color_jitter"] else "") + (
                                                           "blur" if unlab_cfg["blur"] else "")

                        cfg = deepcopy(base_cfg)
                        cfg['general'] = {
                            'tag': tune.grid_search([
                                f"{dataset}_{name}_D{n_subset}{restrict_mode}_S{seed}_{opt}Lr{lr}{blr}{lr_sch}_clip{gclip}_crop{crop[0]}x{crop[1]}bs{batch_size}_flip_r101_dec{dec}_{dec_params}_l{layers[0]}os{output_stride}{'hi' if head_inter else ''}{unlab_str}"])}
                        cfg, load_backbone = decoder_variant(cfg, dec, crop)
                        cfg['model']['backbone_pretraining'] = mono_pretrain if (
                                load_backbone and seg_init != "none") else "imnet"
                        cfg['model']['variant'] = name
                        cfg['model']['depth_pretraining'] = teacher_init
                        cfg['model']['depth_estimator_weights'] = mono_pretrain
                        cfg = setup_optimizer(cfg, opt, lr, blr, None, None, gclip)
                        cfg["training"]["batch_size"] = batch_size
                        cfg = setup_dataset(cfg, dataset, crop, lr_sch)
                        cfg['data']['restrict_to_subset']['mode'] = restrict_mode
                        cfg['data']['restrict_to_subset']['n_subset'] = n_subset
                        if preselect:
                            cfg['data']['restrict_to_subset']['subset'] = preselected_labels(
                                {7: 42, 25: 43, 42: 44}[seed], n_subset, dataset, method=pres_method,
                            )
                        cfg['training']['unlabeled_segmentation'] = unlab_cfg
                        cfg['seed'] = seed
                        cfg = set_segmentation_args(cfg, seg_init=seg_init, layers=layers, head_inter=head_inter,
                                                    output_stride=output_stride)
                        cfgs.append(cfg)
    # Data Selection for Annotation
    elif id == 211:
        layers = [8]
        output_stride = 2
        head_inter = True
        opt = "adam"
        lr = 1e-4
        blr = 1e-4
        plr = 1e-6
        slr = 1e-4
        mono_lambda = 0
        psd_lambda = 1
        seg_lambda = 1
        depth_loss_log = False
        dataset = "cityscapes" # available: cityscapes, camvid, mapillary
        lr_sch = "poly"
        gclip = 100000
        dec, dec_params, crop, batch_size = (9, "", (512, 512), 2)
        for seed in [42, 43, 44]:
            mono_pretrain = f'mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd2_crop512x512bs2'
            last_segmentation_only = True
            label_schedule, label_steps, train_iters_per_step, train_from_scratch, last_from_scratch = {
                "cityscapes": ("labsch_25-50-100-200-372-744_4-8-12-16-20-24-scratch", [25, 50, 100, 200, 372, 744],
                    [4e3, 8e3, 12e3, 16e3, 20e3, 24e3], True, True),
                "camvid": ("labsch_25-50-100_4-8-12-scratch", [25, 50, 100], [4e3, 8e3, 12e3], True, True),
            }[dataset]
            for name, depth_lambda, entropy_lambda, dtype, selection_tasks, choice, depthifp_w, n_pres, bias_w, ifp_args in [
                # Active Learning Segmentation Entropy Selection
                # ('entropy_sonly', 0, 1, "abs", "seg", "score", 0, None, 0, {}),
                # Ours (Diversity Sampling)
                # ("depthifp_u3-avg4", 0, 0, "abs", "depth", "ifp", 1, None, 0,
                #  {'p': 2, 'pool': 'avg', 'h': 4, 'm': 'u3', 'norm': True}),
                # Ours (Uncertainty Sampling)
                # ("ldepth_donly", 1, 0, "abs_log", "depth", "score", 0, None, 0, {}),
                # Ours (Diversity and Uncertainty Sampling)
                ("depthifp_u3-avg4_bias1000ldepth_donly", 1, 0, "abs_log", "depth", "ifp", 1, None, 1000,
                 {'p': 2, 'pool': 'avg', 'h': 4, 'm': 'u3', 'norm': True}),
            ]:
                assert selection_tasks in ["depth", "seg", "seg+depth"]
                cfg = deepcopy(base_cfg)
                cfg['main'] = "label_selection"
                cfg['label_selection'] = {
                    'choice': choice,
                    'label_steps': label_steps,
                    'train_iters': train_iters_per_step,
                    'train_from_scratch': train_from_scratch,
                    'last_from_scratch': last_from_scratch,
                    'selection_tasks': selection_tasks,
                    'last_segmentation_only': last_segmentation_only,
                    'last_depth_only': False,
                    'initial_samples': "ifp" if choice == "ifp" else "random",
                    'preselection_multiplier': n_pres,
                    'depth_ifp_weight': depthifp_w,
                    'bias_weight': bias_w,
                    'ifp_args': ifp_args,
                    'depth_lambda': depth_lambda,
                    'entropy_lambda': entropy_lambda,
                    'depth_error_types': dtype,
                    'remove_models': True,
                    'resume': ifp_args.get("resume", (-1, "")),
                }
                cfg['general'] = {
                    'tag': tune.grid_search([
                        f"{dataset}_{name}_{label_schedule}_{'evseg_' if last_segmentation_only else ''}_S{seed}_{opt}Lr{lr:.1E}{slr:.1E}{blr:.1E}{plr:.1E}{lr_sch}_clip{gclip}_m{mono_lambda}s{seg_lambda}pd{psd_lambda}_dl{depth_loss_log}_crop{crop[0]}x{crop[1]}bs{batch_size}_flip_r101_dec{dec}_{dec_params}_l{layers[0]}os{output_stride}{'hi' if head_inter else ''}"])}
                cfg['model']['backbone_name'] = "resnet50"
                cfg, load_backbone = decoder_variant(cfg, dec, crop)
                cfg['model']['backbone_pretraining'] = "imnet"
                cfg['model']['variant'] = name
                cfg['model']['depth_pretraining'] = "none"
                cfg['model']['pose_pretraining'] = mono_pretrain
                cfg['model']['disable_pose'] = mono_lambda == 0
                cfg['model']['disable_monodepth'] = False
                cfg['training']['segmentation_lambda'] = seg_lambda
                cfg['training']['monodepth_lambda'] = mono_lambda
                cfg['training']['pseudo_depth_lambda'] = psd_lambda
                cfg["data"]["depth_teacher"] = mono_pretrain
                cfg = setup_optimizer(cfg, opt, lr, blr, plr, slr, gclip)
                cfg['training']['pseudo_depth_loss_log'] = depth_loss_log
                cfg["training"]["batch_size"] = batch_size
                cfg = setup_dataset(cfg, dataset, crop, lr_sch)
                cfg['data']['restrict_to_subset'] = None
                if train_from_scratch:
                    train_iters = cfg['label_selection']['train_iters'][-1]
                else:
                    train_iters = sum(cfg['label_selection']['train_iters'])
                cfg['training']['train_iters'] = train_iters
                cfg = lr_schedule(cfg, lr_sch, train_iters)
                cfg['seed'] = seed
                cfg = set_segmentation_args(cfg, seg_init="none", layers=layers, head_inter=head_inter,
                                            output_stride=output_stride, head_dropout=0.0)
                cfgs.append(cfg)
    # Semi-Supervised Segmentation with Multi-Task Learning
    elif id == 212:
        final_layer = 9
        distillation_layer = 7
        output_stride = 1
        side_output = True
        opt = "sgd"
        lr = 1e-2
        blr = 1e-3
        plr = 1e-6
        dlr = 1e-3
        gclip = 10
        disable_depth_clip = False
        dataset = "cityscapes"
        lr_sch = "stepx"
        backward_first_pseudo_label = False
        mono_lambda = 1
        seg_lambda = 1
        dec, dec_params, crop, batch_size = (6, "lr5_fd2_crop512x512bs4", (512, 512), 2)
        for seed in [
            # 7,
            # 25,
            42
        ]:
            for n_subset in subsets(dataset):
                dc_ft = 0
                dc_m = 0.03
                pres_method = "ds_us"  # available: "ent", "ds", "us", "ds_us"
                for name, ema, mix_mask, only_unlabeled, mix_use_gt, preselect in [
                    (f'pad_transfer_dcompgt{dc_m}{dc_ft}', True, "depthcomp", False, True, False),
                    (f'sel_{pres_method}_pad_transfer_dcompgt{dc_m}{dc_ft}', True, "depthcomp", False, True, True),
                ]:
                    name = name.replace('.', '').replace(' ', '').replace(',', 'i').replace('(', 'I').replace(')', 'I')
                    restrict_mode = "fixed" if preselect else "random"
                    unlab_cfg = {
                        "consistency_weight": 1.0,
                        "mix_mask": mix_mask,
                        "depthmix_online_depth": True,
                        "backward_first_pseudo_label": backward_first_pseudo_label,
                        "color_jitter": True,
                        "blur": True,
                        "only_unlabeled": only_unlabeled,
                        "mix_use_gt": mix_use_gt,
                        "depthcomp_margin": dc_m,
                        "depthcomp_foreground_threshold": dc_ft,
                        "debug_image": True
                    } if ema else None
                    unlab_str = "" if not ema else f"_Unlab{unlab_cfg['consistency_weight']}{unlab_cfg['mix_mask']}" + \
                                                   f"FPL{backward_first_pseudo_label}" + \
                                                   ("jit" if unlab_cfg["color_jitter"] else "") + (
                                                       "blur" if unlab_cfg["blur"] else "")

                    mono_pretrain = f'mono_cityscapes_1024x512_r101dil_aspp_dec{dec}_{dec_params}'
                    cfg = deepcopy(base_cfg)

                    cfg['general'] = {
                        'tag': tune.grid_search([
                            f"{dataset}_{name}_D{n_subset}{restrict_mode}_S{seed}_{opt}Lr{lr:.0E}{blr:.0E}{plr:.0E}{dlr:.0E}{lr_sch}_clip{gclip}{disable_depth_clip}_m{mono_lambda}s{seg_lambda}_crop{crop[0]}x{crop[1]}bs{batch_size}_flip_dec{dec}_{dec_params}_l{final_layer}i{distillation_layer}{side_output}os{output_stride}{unlab_str}"])}
                    cfg['model']['segmentation_name'] = 'mtl_pad'
                    cfg['model']['backbone_name'] = f"resnet101"
                    cfg, load_backbone = decoder_variant(cfg, dec, crop)
                    cfg['model']['backbone_pretraining'] = mono_pretrain
                    cfg['model']['variant'] = name
                    cfg['model']['depth_estimator_weights'] = mono_pretrain
                    cfg['model']['depth_pretraining'] = mono_pretrain
                    cfg['model']['pose_pretraining'] = mono_pretrain
                    cfg['model']['disable_pose'] = mono_lambda == 0
                    cfg['model']['disable_monodepth'] = False
                    cfg['training']['segmentation_lambda'] = seg_lambda
                    cfg['training']['monodepth_lambda'] = mono_lambda
                    cfg['training']['disable_depth_estimator'] = True
                    cfg = setup_optimizer(cfg, opt, lr, blr, plr, None, gclip)
                    cfg["training"]["disable_depth_grad_clip"] = disable_depth_clip
                    cfg["training"]["batch_size"] = batch_size
                    cfg = setup_dataset(cfg, dataset, crop, lr_sch)
                    cfg['data']['restrict_to_subset']['mode'] = restrict_mode
                    cfg['data']['restrict_to_subset']['n_subset'] = n_subset
                    if preselect:
                        cfg['data']['restrict_to_subset']['subset'] = preselected_labels(
                            {7: 42, 25: 43, 42: 44}[seed], n_subset, dataset, method=pres_method
                        )
                    cfg['training']['unlabeled_segmentation'] = unlab_cfg
                    cfg['seed'] = seed
                    cfg['model']['segmentation_args'] = {
                        'weights': mono_pretrain,
                        'output_stride': output_stride,
                        'distillation_layer': distillation_layer,
                        'side_output': side_output,
                        'final_layer': final_layer
                    }
                    cfgs.append(cfg)
    # Table 7: Framework component ablation (S: Data Selection, DX: DepthMix, MTL: SDE Multi-Task Learning)
    elif id == 213:
        dataset = "cityscapes"
        pres_method = "ds_us"
        dc_ft, dc_m = 0, 0.03
        mono_pretrain = 'mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd2_crop512x512bs4'
        dec, dec_params, crop, batch_size = (6, "lr5_fd2_crop512x512bs4", (512, 512), 2)

        # (S, DX, MTL): (preselect, ema+DepthMix, use_mtl_arch)
        # Table 7: 8 combinations, 372 labels for all, 2975 only for Baseline/MTL/DX/DX+MTL
        table7_combos = [
            (False, False, False, [372, 2975]),   # 1. Baseline
            (False, False, True, [372, 2975]),    # 2. MTL only
            (False, True, False, [372, 2975]),    # 3. DX only
            (True, False, False, [372]),          # 4. S only (2975 not in table)
            (True, False, True, [372]),           # 5. S+MTL (2975 not in table)
            (True, True, False, [372]),           # 6. S+DX (2975 not in table)
            (False, True, True, [372, 2975]),     # 7. DX+MTL
            (True, True, True, [372]),            # 8. S+DX+MTL (2975 not in table)
        ]

        for combo_idx, (preselect, use_dx, use_mtl, n_subsets) in enumerate(table7_combos):
            for n_subset in n_subsets:
                for seed in [7, 25, 42]:
                    restrict_mode = "fixed" if preselect else "random"
                    # MTL experiments always use EMA (semi-supervised), non-MTL experiments use EMA only if DepthMix is enabled
                    ema = use_mtl or use_dx  # MTL always uses EMA, DX also requires EMA

                    if use_mtl:
                        # exp 212 style: MTL (mtl_pad)
                        final_layer, distillation_layer = 9, 7
                        opt, lr, blr, plr, dlr = "sgd", 1e-2, 1e-3, 1e-6, 1e-3
                        gclip, disable_depth_clip = 10, False
                        mono_lambda, seg_lambda = 1, 1
                        lr_sch, backward_first = "stepx", False

                        mix_mask = "depthcomp" if use_dx else None
                        unlab_cfg = {
                            "consistency_weight": 1.0, "mix_mask": mix_mask, "depthmix_online_depth": use_dx,
                            "backward_first_pseudo_label": backward_first, "color_jitter": True, "blur": True,
                            "only_unlabeled": False, "mix_use_gt": use_dx, "depthcomp_margin": dc_m,
                            "depthcomp_foreground_threshold": dc_ft, "debug_image": True
                        } if ema else None
                        unlab_str = "" if not ema else f"_Unlab1.0{mix_mask}FPL{backward_first}jitblur"

                        name = f'sel_{pres_method}_pad_transfer_dcompgt{dc_m}{dc_ft}' if preselect else f'pad_transfer_dcompgt{dc_m}{dc_ft}'
                        if not use_dx:
                            name = f'sel_{pres_method}_pad_transfer' if preselect else 'pad_transfer'
                        name = name.replace('.', '').replace(' ', '').replace(',', 'i').replace('(', 'I').replace(')', 'I')

                        cfg = deepcopy(base_cfg)
                        cfg['general'] = {
                            'tag': tune.grid_search([
                                f"{dataset}_{name}_D{n_subset}{restrict_mode}_S{seed}_{opt}Lr{lr:.0E}{blr:.0E}{plr:.0E}{dlr:.0E}{lr_sch}_clip{gclip}{disable_depth_clip}_m{mono_lambda}s{seg_lambda}_crop{crop[0]}x{crop[1]}bs{batch_size}_flip_dec{dec}_{dec_params}_l{final_layer}i{distillation_layer}Trueos1{unlab_str}"])}
                        cfg['model']['segmentation_name'] = 'mtl_pad'
                        cfg['model']['backbone_name'] = 'resnet101'
                        cfg, _ = decoder_variant(cfg, dec, crop)
                        cfg['model']['backbone_pretraining'] = mono_pretrain
                        cfg['model']['variant'] = name
                        cfg['model']['depth_estimator_weights'] = mono_pretrain
                        cfg['model']['depth_pretraining'] = mono_pretrain
                        cfg['model']['pose_pretraining'] = mono_pretrain
                        cfg['model']['disable_pose'] = mono_lambda == 0
                        cfg['model']['disable_monodepth'] = False
                        cfg['training']['segmentation_lambda'] = seg_lambda
                        cfg['training']['monodepth_lambda'] = mono_lambda
                        cfg['training']['disable_depth_estimator'] = True
                        cfg = setup_optimizer(cfg, opt, lr, blr, plr, None, gclip)
                        cfg["training"]["disable_depth_grad_clip"] = disable_depth_clip
                        cfg["training"]["batch_size"] = batch_size
                        cfg = setup_dataset(cfg, dataset, crop, lr_sch)
                        cfg['data']['restrict_to_subset']['mode'] = restrict_mode
                        cfg['data']['restrict_to_subset']['n_subset'] = n_subset
                        if preselect:
                            cfg['data']['restrict_to_subset']['subset'] = preselected_labels(
                                {7: 42, 25: 43, 42: 44}[seed], n_subset, dataset, method=pres_method
                            )
                        cfg['training']['unlabeled_segmentation'] = unlab_cfg
                        cfg['seed'] = seed
                        cfg['model']['segmentation_args'] = {
                            'weights': mono_pretrain, 'output_stride': 1,
                            'distillation_layer': distillation_layer, 'side_output': True, 'final_layer': final_layer
                        }
                        cfgs.append(cfg)
                    else:
                        # exp 210 style: joint_seg_depth (transfer-based, with or without DepthMix)
                        layers, output_stride, head_inter = [9], 1, False
                        opt, lr, blr, gclip = "sgd", 1e-2, 1e-3, 10
                        lr_sch = "stepx"

                        # Table 7 baseline and all experiments are transfer-based (not scratch)
                        seg_init, teacher_init = mono_pretrain, mono_pretrain
                        if use_dx:
                            mix_mask = "depthcomp"
                            only_unlabeled, mix_use_gt = False, True
                            name = f'sel_{pres_method}_transfer_dcompgt{dc_m}{dc_ft}' if preselect else f'transfer_dcompgt{dc_m}{dc_ft}'
                        else:
                            mix_mask = None
                            only_unlabeled, mix_use_gt = True, False
                            name = f'sel_{pres_method}_transfer' if preselect else 'transfer'

                        name = name.replace('.', '').replace(' ', '').replace(',', 'i').replace('(', 'I').replace(')', 'I')
                        unlab_cfg = {
                            "consistency_weight": 1.0, "mix_mask": mix_mask, "color_jitter": True, "blur": True,
                            "only_unlabeled": only_unlabeled, "only_labeled": False, "mix_video": False,
                            "mix_use_gt": mix_use_gt, "depthcomp_margin": dc_m, "depthcomp_foreground_threshold": dc_ft,
                            "backward_first_pseudo_label": False, "debug_image": True,
                            "depthmix_online_depth": use_dx,
                        } if ema else None
                        unlab_str = "" if not ema else f"_Unlab{1.0}{mix_mask}jitblur"

                        cfg = deepcopy(base_cfg)
                        cfg['general'] = {
                            'tag': tune.grid_search([
                                f"{dataset}_{name}_D{n_subset}{restrict_mode}_S{seed}_{opt}Lr{lr}{blr}{lr_sch}_clip{gclip}_crop{crop[0]}x{crop[1]}bs{batch_size}_flip_r101_dec{dec}_{dec_params}_l{layers[0]}os{output_stride}{'hi' if head_inter else ''}{unlab_str}"])}
                        cfg, load_backbone = decoder_variant(cfg, dec, crop)
                        cfg['model']['backbone_pretraining'] = mono_pretrain if (load_backbone and seg_init != "none") else "imnet"
                        cfg['model']['variant'] = name
                        cfg['model']['depth_pretraining'] = teacher_init
                        cfg['model']['depth_estimator_weights'] = mono_pretrain
                        if use_dx:
                            cfg['model']['disable_monodepth'] = False
                            cfg['model']['disable_pose'] = False
                            cfg['model']['pose_pretraining'] = mono_pretrain
                            cfg['training']['monodepth_lambda'] = 1
                            cfg['training']['pseudo_depth_lambda'] = 1
                        cfg = setup_optimizer(cfg, opt, lr, blr, None, None, gclip)
                        cfg["training"]["batch_size"] = batch_size
                        cfg = setup_dataset(cfg, dataset, crop, lr_sch)
                        cfg['data']['restrict_to_subset']['mode'] = restrict_mode
                        cfg['data']['restrict_to_subset']['n_subset'] = n_subset
                        if preselect:
                            cfg['data']['restrict_to_subset']['subset'] = preselected_labels(
                                {7: 42, 25: 43, 42: 44}[seed], n_subset, dataset, method=pres_method
                            )
                        cfg['training']['unlabeled_segmentation'] = unlab_cfg
                        cfg['seed'] = seed
                        cfg = set_segmentation_args(cfg, seg_init=seg_init, layers=layers, head_inter=head_inter, output_stride=output_stride)
                        cfgs.append(cfg)
    # exp 219: No-detach projection ablation (single vs dual, cosine/mse, warmup, λ search), 1 seed
    # projection_mode × ct_type × warmup_iters × lambda_ct = 2×2×2×4 = 32 runs. Seed=7.
    # λ 作为超参数在 {0.25, 0.5, 0.75, 1.0} 上搜索
    elif id == 219:
        dataset = "cityscapes"
        mono_pretrain = 'mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd2_crop512x512bs4'
        dec, dec_params, crop, batch_size = (6, "lr5_fd2_crop512x512bs4", (512, 512), 2)
        n_subset = 372
        final_layer, distillation_layer = 9, 7
        opt, lr, blr, plr = "sgd", 1e-2, 1e-3, 1e-6
        gclip = 10
        mono_lambda, seg_lambda = 1, 1
        lr_sch = "stepx"
        seed = 7
        unlab_cfg = {
            "consistency_weight": 1.0, "mix_mask": None, "depthmix_online_depth": False,
            "backward_first_pseudo_label": False, "color_jitter": True, "blur": True,
            "only_unlabeled": False, "mix_use_gt": False, "depthcomp_margin": 0.03,
            "depthcomp_foreground_threshold": 0, "debug_image": True
        }
        lambdas = [0.25, 0.5, 0.75, 1.0]
        for projection_mode in ["single", "dual"]:
            for ct_type in ["cosine", "mse"]:
                for warmup_iters in [5000, 0]:
                    w_tag = "w5k" if warmup_iters == 5000 else "w0"
                    for lam in lambdas:
                        lam_tag = f"lam{lam:.2f}".replace(".", "p")
                        group_label = f"{projection_mode}_{ct_type}_{w_tag}_{lam_tag}"
                        cfg = deepcopy(base_cfg)
                        if cfg['data'].get('restrict_to_subset') is None:
                            cfg['data']['restrict_to_subset'] = {}
                        tag = f"{dataset}_pad_ct_{group_label}_D{n_subset}_S{seed}"
                        cfg['general'] = {'tag': tune.grid_search([tag])}
                        cfg['model']['segmentation_name'] = 'mtl_pad'
                        cfg['model']['backbone_name'] = 'resnet101'
                        cfg, _ = decoder_variant(cfg, dec, crop)
                        cfg['model']['backbone_pretraining'] = mono_pretrain
                        cfg['model']['variant'] = f'ct_{group_label}'
                        cfg['model']['depth_estimator_weights'] = mono_pretrain
                        cfg['model']['depth_pretraining'] = mono_pretrain
                        cfg['model']['pose_pretraining'] = mono_pretrain
                        cfg['model']['disable_pose'] = False
                        cfg['model']['disable_monodepth'] = False
                        cfg['training']['segmentation_lambda'] = seg_lambda
                        cfg['training']['monodepth_lambda'] = mono_lambda
                        cfg['training']['cross_task_lambda'] = lam
                        cfg['training']['cross_task_type'] = ct_type
                        cfg['training']['cross_task_warmup_iters'] = warmup_iters
                        cfg['training']['disable_depth_estimator'] = True
                        cfg = setup_optimizer(cfg, opt, lr, blr, plr, None, gclip)
                        cfg["training"]["disable_depth_grad_clip"] = False
                        cfg["training"]["batch_size"] = batch_size
                        cfg = setup_dataset(cfg, dataset, crop, lr_sch)
                        cfg['data']['restrict_to_subset']['mode'] = 'random'
                        cfg['data']['restrict_to_subset']['n_subset'] = n_subset
                        cfg['training']['unlabeled_segmentation'] = unlab_cfg
                        cfg['seed'] = seed
                        cfg['model']['segmentation_args'] = {
                            'weights': mono_pretrain, 'output_stride': 1,
                            'distillation_layer': distillation_layer, 'side_output': True, 'final_layer': final_layer,
                            'projection_mode': projection_mode,
                        }
                        cfgs.append(cfg)
    # exp 220: 3-seeds 验证 exp219 候选配置（λ 细扫 + MSE 对照）
    # 配置：single_cosine w0 (0.75,1.0,1.25); single_mse w5k/w0 λ=1.0; dual_cosine w0/w5k (0.5,0.75,1.0,1.25)
    # 每个配置 3 seeds (7,25,42)，共 13 组 × 3 = 39 runs
    elif id == 220:
        dataset = "cityscapes"
        mono_pretrain = 'mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd2_crop512x512bs4'
        dec, dec_params, crop, batch_size = (6, "lr5_fd2_crop512x512bs4", (512, 512), 2)
        n_subset = 372
        final_layer, distillation_layer = 9, 7
        opt, lr, blr, plr = "sgd", 1e-2, 1e-3, 1e-6
        gclip = 10
        mono_lambda, seg_lambda = 1, 1
        lr_sch = "stepx"
        seeds = [7, 25, 42]
        unlab_cfg = {
            "consistency_weight": 1.0, "mix_mask": None, "depthmix_online_depth": False,
            "backward_first_pseudo_label": False, "color_jitter": True, "blur": True,
            "only_unlabeled": False, "mix_use_gt": False, "depthcomp_margin": 0.03,
            "depthcomp_foreground_threshold": 0, "debug_image": True
        }
        # (projection_mode, ct_type, warmup_iters, lam) → run_id base
        configs_220 = [
            ("single", "cosine", 0, 0.75),
            ("single", "cosine", 0, 1.0),
            ("single", "cosine", 0, 1.25),
            ("single", "mse", 5000, 1.0),
            ("single", "mse", 0, 1.0),
            ("dual", "cosine", 0, 0.5),
            ("dual", "cosine", 0, 0.75),
            ("dual", "cosine", 0, 1.0),
            ("dual", "cosine", 0, 1.25),
            ("dual", "cosine", 5000, 0.5),
            ("dual", "cosine", 5000, 0.75),
            ("dual", "cosine", 5000, 1.0),
            ("dual", "cosine", 5000, 1.25),
        ]
        for projection_mode, ct_type, warmup_iters, lam in configs_220:
            w_tag = "w5k" if warmup_iters == 5000 else "w0"
            lam_tag = f"lam{lam:.2f}".replace(".", "p")
            group_label = f"{projection_mode}_{ct_type}_{w_tag}_{lam_tag}"
            for seed in seeds:
                cfg = deepcopy(base_cfg)
                if cfg['data'].get('restrict_to_subset') is None:
                    cfg['data']['restrict_to_subset'] = {}
                tag = f"{dataset}_pad_ct_{group_label}_D{n_subset}_S{seed}"
                cfg['general'] = {'tag': tune.grid_search([tag])}
                cfg['model']['segmentation_name'] = 'mtl_pad'
                cfg['model']['backbone_name'] = 'resnet101'
                cfg, _ = decoder_variant(cfg, dec, crop)
                cfg['model']['backbone_pretraining'] = mono_pretrain
                cfg['model']['variant'] = f'ct_{group_label}'
                cfg['model']['depth_estimator_weights'] = mono_pretrain
                cfg['model']['depth_pretraining'] = mono_pretrain
                cfg['model']['pose_pretraining'] = mono_pretrain
                cfg['model']['disable_pose'] = False
                cfg['model']['disable_monodepth'] = False
                cfg['training']['segmentation_lambda'] = seg_lambda
                cfg['training']['monodepth_lambda'] = mono_lambda
                cfg['training']['cross_task_lambda'] = lam
                cfg['training']['cross_task_type'] = ct_type
                cfg['training']['cross_task_warmup_iters'] = warmup_iters
                cfg['training']['disable_depth_estimator'] = True
                cfg = setup_optimizer(cfg, opt, lr, blr, plr, None, gclip)
                cfg["training"]["disable_depth_grad_clip"] = False
                cfg["training"]["batch_size"] = batch_size
                cfg = setup_dataset(cfg, dataset, crop, lr_sch)
                cfg['data']['restrict_to_subset']['mode'] = 'random'
                cfg['data']['restrict_to_subset']['n_subset'] = n_subset
                cfg['training']['unlabeled_segmentation'] = unlab_cfg
                cfg['seed'] = seed
                cfg['model']['segmentation_args'] = {
                    'weights': mono_pretrain, 'output_stride': 1,
                    'distillation_layer': distillation_layer, 'side_output': True, 'final_layer': final_layer,
                    'projection_mode': projection_mode,
                }
                cfgs.append(cfg)

    # exp 221: MTL + DepthMix / MTL + Selection / MTL + DepthMix + Selection
    # 全部 N=372，3 seeds (7, 25, 42)，共 3 组 × 3 = 9 runs
    # 基础设置与 exp213 MTL 分支完全一致
    elif id == 221:
        dataset = "cityscapes"
        pres_method = "ds_us"
        dc_ft, dc_m = 0, 0.03
        mono_pretrain = 'mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd2_crop512x512bs4'
        dec, dec_params, crop, batch_size = (6, "lr5_fd2_crop512x512bs4", (512, 512), 2)
        n_subset = 372
        final_layer, distillation_layer = 9, 7
        opt, lr, blr, plr, dlr = "sgd", 1e-2, 1e-3, 1e-6, 1e-3
        gclip, disable_depth_clip = 10, False
        mono_lambda, seg_lambda = 1, 1
        lr_sch, backward_first = "stepx", False

        # (preselect, use_dx, label) — all use MTL (use_mtl=True)
        exp221_combos = [
            (False, True,  "mtl_dx"),    # run 0-2:  MTL + DepthMix
            (True,  False, "mtl_sel"),   # run 3-5:  MTL + Selection
            (True,  True,  "mtl_dx_sel"),# run 6-8:  MTL + DepthMix + Selection
        ]

        for preselect, use_dx, combo_label in exp221_combos:
            restrict_mode = "fixed" if preselect else "random"
            mix_mask = "depthcomp" if use_dx else None
            unlab_cfg = {
                "consistency_weight": 1.0, "mix_mask": mix_mask,
                "depthmix_online_depth": use_dx,
                "backward_first_pseudo_label": backward_first,
                "color_jitter": True, "blur": True,
                "only_unlabeled": False, "mix_use_gt": use_dx,
                "depthcomp_margin": dc_m,
                "depthcomp_foreground_threshold": dc_ft,
                "debug_image": True,
            }
            if use_dx:
                name_base = f'sel_{pres_method}_pad_transfer_dcompgt{dc_m}{dc_ft}' if preselect \
                            else f'pad_transfer_dcompgt{dc_m}{dc_ft}'
            else:
                name_base = f'sel_{pres_method}_pad_transfer' if preselect else 'pad_transfer'
            name_base = name_base.replace('.', '').replace(' ', '')
            unlab_str = f"_Unlab1.0{mix_mask}FPL{backward_first}jitblur"

            for seed in [7, 25, 42]:
                cfg = deepcopy(base_cfg)
                cfg['general'] = {
                    'tag': tune.grid_search([
                        f"{dataset}_{name_base}_D{n_subset}{restrict_mode}_S{seed}_{opt}Lr{lr:.0E}{blr:.0E}{plr:.0E}{dlr:.0E}{lr_sch}_clip{gclip}{disable_depth_clip}_m{mono_lambda}s{seg_lambda}_crop{crop[0]}x{crop[1]}bs{batch_size}_flip_dec{dec}_{dec_params}_l{final_layer}i{distillation_layer}Trueos1{unlab_str}"
                    ])}
                cfg['model']['segmentation_name'] = 'mtl_pad'
                cfg['model']['backbone_name'] = 'resnet101'
                cfg, _ = decoder_variant(cfg, dec, crop)
                cfg['model']['backbone_pretraining'] = mono_pretrain
                cfg['model']['variant'] = name_base
                cfg['model']['depth_estimator_weights'] = mono_pretrain
                cfg['model']['depth_pretraining'] = mono_pretrain
                cfg['model']['pose_pretraining'] = mono_pretrain
                cfg['model']['disable_pose'] = mono_lambda == 0
                cfg['model']['disable_monodepth'] = False
                cfg['training']['segmentation_lambda'] = seg_lambda
                cfg['training']['monodepth_lambda'] = mono_lambda
                cfg['training']['disable_depth_estimator'] = True
                cfg = setup_optimizer(cfg, opt, lr, blr, plr, None, gclip)
                cfg["training"]["disable_depth_grad_clip"] = disable_depth_clip
                cfg["training"]["batch_size"] = batch_size
                cfg = setup_dataset(cfg, dataset, crop, lr_sch)
                cfg['data']['restrict_to_subset']['mode'] = restrict_mode
                cfg['data']['restrict_to_subset']['n_subset'] = n_subset
                if preselect:
                    cfg['data']['restrict_to_subset']['subset'] = preselected_labels(
                        {7: 42, 25: 43, 42: 44}[seed], n_subset, dataset, method=pres_method
                    )
                cfg['training']['unlabeled_segmentation'] = unlab_cfg
                cfg['seed'] = seed
                cfg['model']['segmentation_args'] = {
                    'weights': mono_pretrain, 'output_stride': 1,
                    'distillation_layer': distillation_layer,
                    'side_output': True, 'final_layer': final_layer,
                }
                cfgs.append(cfg)

    # exp 222: 完整模型 = exp221 三组配置 + cross-task projection loss (single, w0, λ=1.0)
    # 两种 loss 类型都测试：MSE λ=1.0 和 Cosine λ=1.0，均无 warm-up
    # projection_mode='single'
    # Run IDs:
    #   MSE    λ=1.0: 0-2  = MTL+DX,  3-5  = MTL+Sel,  6-8  = MTL+DX+Sel
    #   Cosine λ=1.0: 9-11 = MTL+DX, 12-14 = MTL+Sel, 15-17 = MTL+DX+Sel
    elif id == 222:
        dataset = "cityscapes"
        pres_method = "ds_us"
        dc_ft, dc_m = 0, 0.03
        mono_pretrain = 'mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd2_crop512x512bs4'
        dec, dec_params, crop, batch_size = (6, "lr5_fd2_crop512x512bs4", (512, 512), 2)
        n_subset = 372
        final_layer, distillation_layer = 9, 7
        opt, lr, blr, plr, dlr = "sgd", 1e-2, 1e-3, 1e-6, 1e-3
        gclip, disable_depth_clip = 10, False
        mono_lambda, seg_lambda = 1, 1
        lr_sch, backward_first = "stepx", False
        projection_mode = "single"

        # (ct_type, ct_lambda) — both w0 (no warm-up), λ=1.0
        ct_variants = [
            ("mse",    1.0),  # runs  0- 8
            ("cosine", 1.0),  # runs  9-17
        ]

        exp222_combos = [
            (False, True,  "mtl_dx"),     # +0: MTL + DX
            (True,  False, "mtl_sel"),    # +3: MTL + Sel
            (True,  True,  "mtl_dx_sel"), # +6: MTL + DX + Sel
        ]

        for ct_type, ct_lambda in ct_variants:
            lam_str = f"{ct_lambda:.2f}".replace(".", "p")
            ct_tag = f"ct{ct_type}w0lam{lam_str}"
            for preselect, use_dx, combo_label in exp222_combos:
                restrict_mode = "fixed" if preselect else "random"
                mix_mask = "depthcomp" if use_dx else None
                unlab_cfg = {
                    "consistency_weight": 1.0, "mix_mask": mix_mask,
                    "depthmix_online_depth": use_dx,
                    "backward_first_pseudo_label": backward_first,
                    "color_jitter": True, "blur": True,
                    "only_unlabeled": False, "mix_use_gt": use_dx,
                    "depthcomp_margin": dc_m,
                    "depthcomp_foreground_threshold": dc_ft,
                    "debug_image": True,
                }
                if use_dx:
                    name_base = f'sel_{pres_method}_pad_transfer_dcompgt{dc_m}{dc_ft}_{ct_tag}' if preselect \
                                else f'pad_transfer_dcompgt{dc_m}{dc_ft}_{ct_tag}'
                else:
                    name_base = f'sel_{pres_method}_pad_transfer_{ct_tag}' if preselect \
                                else f'pad_transfer_{ct_tag}'
                name_base = name_base.replace('.', '').replace(' ', '')
                unlab_str = f"_Unlab1.0{mix_mask}FPL{backward_first}jitblur"

                for seed in [7, 25, 42]:
                    cfg = deepcopy(base_cfg)
                    cfg['general'] = {
                        'tag': tune.grid_search([
                            f"{dataset}_{name_base}_D{n_subset}{restrict_mode}_S{seed}_{opt}Lr{lr:.0E}{blr:.0E}{plr:.0E}{dlr:.0E}{lr_sch}_clip{gclip}{disable_depth_clip}_m{mono_lambda}s{seg_lambda}_crop{crop[0]}x{crop[1]}bs{batch_size}_flip_dec{dec}_{dec_params}_l{final_layer}i{distillation_layer}Trueos1{unlab_str}"
                        ])}
                    cfg['model']['segmentation_name'] = 'mtl_pad'
                    cfg['model']['backbone_name'] = 'resnet101'
                    cfg, _ = decoder_variant(cfg, dec, crop)
                    cfg['model']['backbone_pretraining'] = mono_pretrain
                    cfg['model']['variant'] = name_base
                    cfg['model']['depth_estimator_weights'] = mono_pretrain
                    cfg['model']['depth_pretraining'] = mono_pretrain
                    cfg['model']['pose_pretraining'] = mono_pretrain
                    cfg['model']['disable_pose'] = mono_lambda == 0
                    cfg['model']['disable_monodepth'] = False
                    cfg['training']['segmentation_lambda'] = seg_lambda
                    cfg['training']['monodepth_lambda'] = mono_lambda
                    cfg['training']['disable_depth_estimator'] = True
                    cfg['training']['cross_task_lambda'] = ct_lambda
                    cfg['training']['cross_task_type'] = ct_type
                    cfg['training']['cross_task_warmup_iters'] = 0
                    cfg = setup_optimizer(cfg, opt, lr, blr, plr, None, gclip)
                    cfg["training"]["disable_depth_grad_clip"] = disable_depth_clip
                    cfg["training"]["batch_size"] = batch_size
                    cfg = setup_dataset(cfg, dataset, crop, lr_sch)
                    cfg['data']['restrict_to_subset']['mode'] = restrict_mode
                    cfg['data']['restrict_to_subset']['n_subset'] = n_subset
                    if preselect:
                        cfg['data']['restrict_to_subset']['subset'] = preselected_labels(
                            {7: 42, 25: 43, 42: 44}[seed], n_subset, dataset, method=pres_method
                        )
                    cfg['training']['unlabeled_segmentation'] = unlab_cfg
                    cfg['seed'] = seed
                    cfg['model']['segmentation_args'] = {
                        'weights': mono_pretrain, 'output_stride': 1,
                        'distillation_layer': distillation_layer,
                        'side_output': True, 'final_layer': final_layer,
                        'projection_mode': projection_mode,
                    }
                    cfgs.append(cfg)

    else:
        raise NotImplementedError("Unknown id {}".format(id))

    return cfgs