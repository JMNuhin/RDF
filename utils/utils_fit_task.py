"""
Updated training loop with task-guided contrastive learning.
"""
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.utils import get_lr
from utils.task_losses import FeatureAlignmentLoss, InfoNCEContrastiveLoss, compute_task_guided_loss


def fit_one_epoch_task_guided(
    model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer,
    epoch, epoch_step, gen, Epoch, cuda, fp16, scaler, save_period, save_dir,
    lambda_min=0.05, lambda_max=0.20, beta=0.1, temperature=0.2,
    use_task_losses=True, local_rank=0
):
    """
    Training loop with task-guided feature-centric supervision.
    
    New Args:
        lambda_min: minimum adaptive λ
        lambda_max: maximum adaptive λ
        beta: weight for contrastive loss
        temperature: temperature for InfoNCE
        use_task_losses: whether to use new task losses (False for warmup/ablation)
    """
    loss        = 0
    Dehazy_loss = 0
    loss_detection = 0
    loss_alignment = 0
    loss_contrastive = 0
    lambda_avg = 0
    
    criterion = nn.MSELoss()
    
    # Initialize task-specific loss functions
    if use_task_losses:
        align_criterion = FeatureAlignmentLoss()
        contrast_criterion = InfoNCEContrastiveLoss(temperature=temperature)
    
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
        model_train.train()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        
        # Unpack batch (supports both dual and single fog modes)
        if len(batch) == 4:
            # Dual fog mode: (view1, view2, targets, clean)
            images_v1, images_v2, targets, clean = batch
            dual_fog_mode = True
        else:
            # Single fog mode: (images, targets, clean)
            images_v1, targets, clean = batch
            images_v2 = None
            dual_fog_mode = False
        
        with torch.no_grad():
            if cuda:
                images_v1 = images_v1.cuda(local_rank)
                targets = targets.cuda(local_rank)
                clean = clean.cuda(local_rank)
                if dual_fog_mode:
                    images_v2 = images_v2.cuda(local_rank)
        
        optimizer.zero_grad()

        if not fp16:
            if use_task_losses and dual_fog_mode:
                # Task-guided mode with dual fog views
                # Concatenate view1 and view2 for batch processing
                batch_size = images_v1.size(0)
                combined_input = torch.cat([images_v1, images_v2, clean], dim=0)  # (3B, C, H, W)
                
                # Forward pass
                outputs = model_train(combined_input)
                
                # Split outputs
                detections = outputs['detections']  # [out0, out1, out2]
                dehazing = outputs['dehazing']  # (3B, ...)
                neck_features = outputs['neck_features']  # [(3B, C, H, W), ...]
                severity = outputs['severity']  # (3B, 1)
                spatial_weights = outputs['spatial_weights']  # [(3B, 1, H, W), ...]
                
                # Split features by view
                neck_features_v1 = [f[:batch_size] for f in neck_features]
                neck_features_v2 = [f[batch_size:2*batch_size] for f in neck_features]
                neck_features_clean = [f[2*batch_size:] for f in neck_features]
                
                severity_v1 = severity[:batch_size]
                spatial_weights_v1 = [w[:batch_size] for w in spatial_weights]
                
                # Detection loss (only on view1)
                detect_outputs_v1 = [d[:batch_size] for d in detections]
                loss_value_det = yolo_loss(detect_outputs_v1, targets, images_v1)
                
                # Dehazing loss (on all views if available)
                dehazing_v1 = dehazing[:batch_size]
                loss_dehazy = criterion(dehazing_v1, clean)
                
                # Task-guided losses
                # L_align: align view1 features to clean features (supervised)
                L_align = align_criterion(neck_features_v1, neck_features_clean, spatial_weights_v1)
                
                # L_con: contrastive between view1 and view2
                L_con = contrast_criterion(neck_features_v1, neck_features_v2)
                
                # Adaptive λ(x)
                from nets.task_modules import compute_adaptive_lambda
                lambda_adaptive = compute_adaptive_lambda(severity_v1, lambda_min, lambda_max)
                lambda_mean = lambda_adaptive.mean()
                
                # Total loss
                loss_value = loss_value_det + 0.1 * loss_dehazy + lambda_mean * L_align + beta * L_con
                
                # Track for logging
                loss_alignment += L_align.item()
                loss_contrastive += L_con.item()
                lambda_avg += lambda_mean.item()
            else:
                # Legacy mode (original RDFNet)
                hazy_and_clear = torch.cat([images_v1, clean], dim=0)
                outputs = model_train(hazy_and_clear)
                
                if isinstance(outputs, dict):
                    # New model format
                    detect_outputs = outputs['detections']
                    dehazing = outputs['dehazing']
                    batch_size = images_v1.size(0)
                    detect_outputs_v1 = [d[:batch_size] for d in detect_outputs]
                    dehazing_v1 = dehazing[:batch_size]
                else:
                    # Old model format
                    detect_outputs_v1 = [outputs[0], outputs[1], outputs[2]]
                    dehazing_v1 = outputs[3]
                
                loss_value_det = yolo_loss(detect_outputs_v1, targets, images_v1)
                loss_dehazy = criterion(dehazing_v1, clean)
                loss_value = 1 * loss_value_det + 0.1 * loss_dehazy
            
            loss_value.backward()
            optimizer.step()
        else:
            # FP16 mode
            from torch.amp import autocast
            with autocast('cuda'):
                if use_task_losses and dual_fog_mode:
                    # Same as above but with autocast
                    batch_size = images_v1.size(0)
                    combined_input = torch.cat([images_v1, images_v2, clean], dim=0)
                    
                    outputs = model_train(combined_input)
                    
                    detections = outputs['detections']
                    dehazing = outputs['dehazing']
                    neck_features = outputs['neck_features']
                    severity = outputs['severity']
                    spatial_weights = outputs['spatial_weights']
                    
                    neck_features_v1 = [f[:batch_size] for f in neck_features]
                    neck_features_v2 = [f[batch_size:2*batch_size] for f in neck_features]
                    neck_features_clean = [f[2*batch_size:] for f in neck_features]
                    
                    severity_v1 = severity[:batch_size]
                    spatial_weights_v1 = [w[:batch_size] for w in spatial_weights]
                    
                    detect_outputs_v1 = [d[:batch_size] for d in detections]
                    loss_value_det = yolo_loss(detect_outputs_v1, targets, images_v1)
                    
                    dehazing_v1 = dehazing[:batch_size]
                    loss_dehazy = criterion(dehazing_v1, clean)
                    
                    L_align = align_criterion(neck_features_v1, neck_features_clean, spatial_weights_v1)
                    L_con = contrast_criterion(neck_features_v1, neck_features_v2)
                    
                    from nets.task_modules import compute_adaptive_lambda
                    lambda_adaptive = compute_adaptive_lambda(severity_v1, lambda_min, lambda_max)
                    lambda_mean = lambda_adaptive.mean()
                    
                    loss_value = loss_value_det + 0.1 * loss_dehazy + lambda_mean * L_align + beta * L_con
                    
                    loss_alignment += L_align.item()
                    loss_contrastive += L_con.item()
                    lambda_avg += lambda_mean.item()
                else:
                    hazy_and_clear = torch.cat([images_v1, clean], dim=0)
                    outputs = model_train(hazy_and_clear)
                    
                    if isinstance(outputs, dict):
                        detect_outputs = outputs['detections']
                        dehazing = outputs['dehazing']
                        batch_size = images_v1.size(0)
                        detect_outputs_v1 = [d[:batch_size] for d in detect_outputs]
                        dehazing_v1 = dehazing[:batch_size]
                    else:
                        detect_outputs_v1 = [outputs[0], outputs[1], outputs[2]]
                        dehazing_v1 = outputs[3]
                    
                    loss_value_det = yolo_loss(detect_outputs_v1, targets, images_v1)
                    loss_dehazy = criterion(dehazing_v1, clean)
                    loss_value = 1 * loss_value_det + 0.1 * loss_dehazy
            
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        
        if ema:
            ema.update(model_train)
        
        Dehazy_loss += loss_dehazy.item()
        loss += loss_value.item()
        loss_detection = (loss - 0.1 * Dehazy_loss)
        
        if local_rank == 0:
            postfix_dict = {
                'loss': loss / (iteration + 1),
                'loss_det': loss_detection / (iteration + 1),
                'dehazy': Dehazy_loss / (iteration + 1),
                'lr': get_lr(optimizer)
            }
            
            if use_task_losses and dual_fog_mode:
                postfix_dict.update({
                    'align': loss_alignment / (iteration + 1),
                    'contr': loss_contrastive / (iteration + 1),
                    'λ': lambda_avg / (iteration + 1)
                })
            
            pbar.set_postfix(**postfix_dict)
            pbar.update(1)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, loss / epoch_step)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f' % (loss / epoch_step))
        if use_task_losses and dual_fog_mode:
            print('  Detection: %.3f | Dehazy: %.3f | Align: %.3f | Contrast: %.3f | λ_avg: %.3f' % (
                loss_detection / epoch_step,
                Dehazy_loss / epoch_step,
                loss_alignment / epoch_step,
                loss_contrastive / epoch_step,
                lambda_avg / epoch_step
            ))
        
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()
        
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f.pth" % (epoch + 1, loss / epoch_step)))
        
        if loss / epoch_step <= min(loss_history.losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
        
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))


# Keep original function for backward compatibility
fit_one_epoch = fit_one_epoch_task_guided
