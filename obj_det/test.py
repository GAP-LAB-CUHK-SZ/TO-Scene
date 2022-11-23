from utils.train_test_utils import load_device, get_model, get_dataloader,CheckpointIO,get_tester

def run(cfg):

    '''Load save path'''
    cfg.log_string('Data save path: %s' % (cfg.save_path))
    checkpoint=CheckpointIO(cfg)

    '''Load device'''
    cfg.log_string('Loading device settings.')
    device = load_device(cfg)

    '''Load data'''
    cfg.log_string('Loading dataset.')
    loader = get_dataloader(cfg.config, mode='val')
    #test_loader = get_dataloader(cfg.config, mode='test')

    '''Load net'''
    cfg.log_string('Loading model.')
    net = get_model(cfg.config, device=device).cuda().float()
    checkpoint.register_modules(net=net)

    '''Load trainer'''
    cfg.log_string('Loading tester.')
    tester = get_tester(cfg.config)

    '''Start to train'''
    cfg.log_string('Start to test.')
    tester(cfg, net,loader=loader,checkpoint=checkpoint)

    cfg.log_string('Testing finished.')