import os
import csv
import logging
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, training_args):
        self.args = training_args
        self.writer = None
        self.csv_files = {}
        self.csv_writers = {}

    def setup_logging(self):
        """Setup TensorBoard and CSV logging with hyperparameters in folder name"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Create folder name with hyperparameters
        log_dir = Path(self.args.fout) / f"{timestamp}"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Setup file logging
        log_file = log_dir / 'train.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to the root logger and EX-KG logger
        logging.getLogger().addHandler(file_handler)
        logging.getLogger('EX-KG').addHandler(file_handler)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir)

        # Log hyperparameters
        hparams = vars(self.args)
        self.writer.add_text('hyperparameters', str(hparams), 0)
        
        # Setup CSV files
        csv_dir = log_dir / 'metrics'
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        if self.args.mode == 'rank':
            self._setup_ranking_csv(csv_dir)
        else:
            self._setup_lp_csv(csv_dir)
        
        # Save hyperparameters to JSON
        import json
        with open(os.path.join(log_dir, 'config.json'), 'w') as f:
            json.dump(hparams, f, indent=2)
        
        return log_dir

    def _setup_ranking_csv(self, csv_dir):
        """Setup CSV files for ranking experiments"""
        # Training metrics
        train_file = open(os.path.join(csv_dir, 'train_metrics.csv'), 'w', newline='')
        self.csv_files['train'] = train_file
        self.csv_writers['train'] = csv.DictWriter(
            train_file, 
            fieldnames=['epoch', 'time', 'loss', 'violations']
        )
        self.csv_writers['train'].writeheader()
        
        # Validation metrics
        valid_file = open(os.path.join(csv_dir, 'valid_metrics.csv'), 'w', newline='')
        self.csv_files['valid'] = valid_file
        self.csv_writers['valid'] = csv.DictWriter(
            valid_file,
            fieldnames=['epoch', 'mrr', 'fmrr', 'mean_rank', 'fmean_rank', 'hits10', 'fhits10']
        )
        self.csv_writers['valid'].writeheader()
        
        # Test metrics
        test_file = open(os.path.join(csv_dir, 'test_metrics.csv'), 'w', newline='')
        self.csv_files['test'] = test_file
        self.csv_writers['test'] = csv.DictWriter(
            test_file,
            fieldnames=['epoch', 'mrr', 'fmrr', 'mean_rank', 'fmean_rank', 'hits10', 'fhits10']
        )
        self.csv_writers['test'].writeheader()

    def _setup_lp_csv(self, csv_dir):
        """Setup CSV files for link prediction experiments"""
        train_file = open(os.path.join(csv_dir, 'train_metrics.csv'), 'w', newline='')
        self.csv_files['train'] = train_file
        self.csv_writers['train'] = csv.DictWriter(
            train_file,
            fieldnames=['epoch', 'time', 'loss', 'violations']
        )
        self.csv_writers['train'].writeheader()
        
        valid_file = open(os.path.join(csv_dir, 'valid_metrics.csv'), 'w', newline='')
        self.csv_files['valid'] = valid_file
        self.csv_writers['valid'] = csv.DictWriter(
            valid_file,
            fieldnames=['epoch', 'auc_pr', 'auc_roc']
        )
        self.csv_writers['valid'].writeheader()
        
        test_file = open(os.path.join(csv_dir, 'test_metrics.csv'), 'w', newline='')
        self.csv_files['test'] = test_file
        self.csv_writers['test'] = csv.DictWriter(
            test_file,
            fieldnames=['epoch', 'auc_pr', 'auc_roc']
        )
        self.csv_writers['test'].writeheader()

    def log_train_metrics(self, epoch, time_elapsed, loss=None, violations=None):
        """Log training metrics to TensorBoard and CSV"""
        metrics = {'Time': time_elapsed}
        csv_data = {'time': time_elapsed}
        
        if loss is not None:
            metrics['Loss'] = loss
            csv_data['loss'] = loss
        if violations is not None:
            metrics['Violations'] = violations
            csv_data['violations'] = violations
        
        # TensorBoard
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
        
        # CSV
        if 'train' in self.csv_writers:
            csv_data['epoch'] = epoch
            self.csv_writers['train'].writerow(csv_data)
            self.csv_files['train'].flush()
    
    def log_ranking_metrics(self, phase, epoch, metrics_dict):
        """Log ranking metrics (MRR, Mean Rank, Hits@10) to TensorBoard and CSV"""
        # TensorBoard
        if self.writer:
            self.writer.add_scalar(f'{phase}/MRR', metrics_dict['mrr'], epoch)
            self.writer.add_scalar(f'{phase}/Filtered_MRR', metrics_dict['fmrr'], epoch)
            self.writer.add_scalar(f'{phase}/Mean_Rank', metrics_dict['mean_pos'], epoch)
            self.writer.add_scalar(f'{phase}/Filtered_Mean_Rank', metrics_dict['fmean_pos'], epoch)
            self.writer.add_scalar(f'{phase}/Hits@10', metrics_dict['hits'], epoch)
            self.writer.add_scalar(f'{phase}/Filtered_Hits@10', metrics_dict['fhits'], epoch)
        
        # CSV
        phase_lower = phase.lower()
        if phase_lower in self.csv_writers:
            self.csv_writers[phase_lower].writerow({
                'epoch': epoch,
                'mrr': metrics_dict['mrr'],
                'fmrr': metrics_dict['fmrr'],
                'mean_rank': metrics_dict['mean_pos'],
                'fmean_rank': metrics_dict['fmean_pos'],
                'hits10': metrics_dict['hits'],
                'fhits10': metrics_dict['fhits']
            })
            self.csv_files[phase_lower].flush()
    
    def log_lp_metrics(self, phase, epoch, auc_pr, auc_roc):
        """Log link prediction metrics (AUC-PR, AUC-ROC) to TensorBoard and CSV"""
        # TensorBoard
        if self.writer:
            self.writer.add_scalar(f'{phase}/AUC_PR', auc_pr, epoch)
            self.writer.add_scalar(f'{phase}/AUC_ROC', auc_roc, epoch)
        
        # CSV
        phase_lower = phase.lower()
        if phase_lower in self.csv_writers:
            self.csv_writers[phase_lower].writerow({
                'epoch': epoch,
                'auc_pr': auc_pr,
                'auc_roc': auc_roc
            })
            self.csv_files[phase_lower].flush()

    def close(self):
        """Close all logging resources"""
        if self.writer:
            self.writer.close()
        for f in self.csv_files.values():
            f.close()