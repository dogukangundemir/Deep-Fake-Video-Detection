# central hyper-parameters
BACKBONE       = "efficientnet_b0"      
IMG_SIZE       = 224
CLIP_LEN       = 1                       
NUM_FRAMES     = 8                       
MIXUP_A    = 0.4
LBL_SMOOTH   = 0.05
DROPOUT_P      = 0.3
DROPOUT    = 0.3 
LR             = 3e-4
EPOCHS         = 15
BATCH_SIZE     = 32
AMP            = True                    
SEED           = 42

FREEZE_EPOCHS = 3   
UNFROZEN_LR   = 1e-4