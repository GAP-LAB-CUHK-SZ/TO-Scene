from utils.train_test_utils import get_optimizer, load_scheduler, load_device, get_model, get_dataloader,get_trainer,CheckpointIO
import torch.nn as nn
def run(cfg):

    '''Load save path'''
    cfg.log_string('Data save path: %s' % (cfg.save_path))
    checkpoint=CheckpointIO(cfg)

    '''Load device'''
    cfg.log_string('Loading device settings.')
    device = load_device(cfg)

    '''Load data'''
    cfg.log_string('Loading dataset.')
    train_loader = get_dataloader(cfg.config, mode='train')
    test_loader = get_dataloader(cfg.config, mode='val')

    '''Load net'''
    cfg.log_string('Loading model.')
    net = get_model(cfg.config, device=device)
    '''Load optimizer'''
    cfg.log_string('Loading optimizer.')
    optimizer = get_optimizer(cfg.config, net=net)
    #model, optimizer = amp.initialize(net, optimizer, opt_level="O1")
    checkpoint.register_modules(opt=optimizer)
    net=nn.DataParallel(net)
    checkpoint.register_modules(net=net)

    '''Load scheduler'''
    cfg.log_string('Loading optimizer scheduler.')
    scheduler = load_scheduler(cfg.config, optimizer=optimizer, train_loader=train_loader)
    checkpoint.register_modules(sch=scheduler)

    '''Load trainer'''
    cfg.log_string('Loading trainer.')
    trainer = get_trainer(cfg.config)

    '''Start to train'''
    cfg.log_string('Start to train.')
    trainer(cfg, net, optimizer,scheduler,train_loader=train_loader, test_loader=test_loader,checkpoint=checkpoint)

    cfg.log_string('Training finished.')