from torch import optim

"""
    optimizer_load

    * optimizer 
    * lr scheduler 

"""

def OptimizerTool(name, model, args):
    # optimizer: update the params of model with a strategy to reduce the loss
    
    if name == 'RMSprop':
    #  x(t) = (1-weight_decay) * x(t-1) - [lr/(sqr(r(t))+eps)] * [ g(t) + g(t-1) * momentum]   
    #  r(t) = alpha * r(t-1) + (1-alpha) * g(t) ^2 , r(0) = 0
        optimizer = optim.RMSprop(model.parameters(),
                            lr = args.lr,    # original lr , is not fixed
                            alpha = 0.9, 
                            weight_decay = 0,  # prevent overfitting
                            momentum =0.99
                            )

    if name == 'Adam':
        optimizer = optim.Adam(model.parameters(), 
                               lr=args.lr, 
                               betas=(0.9,0.999), 
                               weight_decay= 1e-8)
        
    if name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), 
                               lr=args.lr, 
                               betas=(0.9,0.999), 
                               weight_decay= 1e-3)
    
    return optimizer


def lrSchedulerTool(name, args, opt):
    
    if name == 'Plateau':
        # if the loss doesn't decrease for [patience] consecutive times, lr = lr * factor
        scheduler = optim.lr_scheduler.ReduceLROnPlateau( optimizer = opt, 
                                                      factor = 0.05 , 
                                                      mode = 'min',  # minimize the loss
                                                      patience = 4 )
    if name == 'Cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max = args.epoch, eta_min=0)

    if name == 'Step':
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.1)

    return scheduler