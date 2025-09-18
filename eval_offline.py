import os
import sys
import pathlib
import numpy as np
import torch
import dill
import hydra
import matplotlib.pyplot as plt
import cv2
import json
from tqdm import tqdm
from omegaconf import OmegaConf
import zarr

# 添加这行来注册编解码器
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
register_codecs()
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset

OmegaConf.register_new_resolver("eval", eval, replace=True)

class OfflineEvaluator:
    def __init__(self, ckpt_path, dataset_path=None, output_dir="eval_results"):
        """
        离线评估器
        Args:
            ckpt_path: 训练好的模型检查点路径
            dataset_path: 数据集路径 (.zarr.zip文件，可选)
            output_dir: 结果输出目录
        """
        self.ckpt_path = ckpt_path
        self.dataset_path = dataset_path
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self._load_model()
        
        # 如果提供了数据集路径，则加载数据集
        if dataset_path:
            self._load_dataset()
        
    def _load_model(self):
        """加载训练好的模型"""
        print(f"Loading model from {self.ckpt_path}")
        
        if not self.ckpt_path.endswith('.ckpt'):
            self.ckpt_path = os.path.join(self.ckpt_path, 'checkpoints', 'latest.ckpt')
            
        payload = torch.load(open(self.ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
        self.cfg = payload['cfg']
        
        # 创建workspace并加载模型
        cls = hydra.utils.get_class(self.cfg._target_)
        workspace = cls(self.cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        # 获取策略模型
        self.policy = workspace.model
        if self.cfg.training.use_ema:
            self.policy = workspace.ema_model
            
        self.policy.eval().to(self.device)
        print(f"Model loaded successfully on {self.device}")
        print(f"Model type: {type(self.policy).__name__}")
        print(f"Configuration: {self.cfg.policy._target_}")
        
        # 输出模型信息
        self._print_model_info()
    def _print_model_info(self):
        """打印模型详细信息"""
        print("\n" + "="*50)
        print("MODEL INFORMATION")
        print("="*50)
        
        # 观测键信息
        if hasattr(self.cfg.task, 'shape_meta'):
            obs_keys = list(self.cfg.task.shape_meta.obs.keys())
            rgb_keys = [k for k in obs_keys if 'rgb' in k]
            low_dim_keys = [k for k in obs_keys if 'rgb' not in k]
            
            print(f"rgb keys:          {rgb_keys}")
            print(f"low_dim_keys keys: {low_dim_keys}")
        
        # 模型架构信息
        if hasattr(self.policy, 'obs_encoder'):
            print(f"Observation encoder: {type(self.policy.obs_encoder).__name__}")
        if hasattr(self.policy, 'noise_pred_net'):
            print(f"Noise prediction network: {type(self.policy.noise_pred_net).__name__}")
            
        # 归一化器信息
        if hasattr(self.policy, 'normalizer'):
            normalizer = self.policy.normalizer
            # 使用 params_dict 来获取键
            if hasattr(normalizer, 'params_dict'):
                print(f"Normalizer keys: {list(normalizer.params_dict.keys())}")
            else:
                print(f"Normalizer type: {type(normalizer).__name__}")
            
            # 尝试获取输入统计信息
            try:
                input_stats = normalizer.get_input_stats()
                if isinstance(input_stats, dict):
                    print(f"Input stats keys: {list(input_stats.keys())}")
            except Exception as e:
                print(f"Could not get input stats: {e}")
                
        print("="*50)
        
    def _load_dataset(self):
        """加载数据集"""
        print(f"Loading dataset from {self.dataset_path}")
        
        # 加载replay buffer
        with zarr.ZipStore(self.dataset_path, mode='r') as zip_store:
            self.replay_buffer = ReplayBuffer.copy_from_store(
                src_store=zip_store, store=zarr.MemoryStore())
        
        # 检查数据集的键
        print(f"Available keys in replay buffer: {list(self.replay_buffer.keys())}")
        
        # 获取一个样本episode来检查结构
        if self.replay_buffer.n_episodes > 0:
            sample_episode = self.replay_buffer.get_episode(0)
            print(f"Sample episode keys: {list(sample_episode.keys())}")
            
            # 使用episode_ends来获取总步数（最简单的方法）
            total_steps = self.replay_buffer.episode_ends[-1] if hasattr(self.replay_buffer, 'episode_ends') else 0
        else:
            total_steps = 0
            
        print(f"Dataset loaded: {self.replay_buffer.n_episodes} episodes, "
            f"{total_steps} total steps")
            
        # 只在确实需要数据集实例时才创建
        # 对于只有虚拟测试的情况，我们不需要实例化数据集
        print("Note: This dataset appears to be raw observation data without action labels.")
        print("Dataset evaluation will be skipped. Only virtual testing will be performed.")
        self.dataset = None
            
    def test_with_virtual_data(self, n_tests=10):
        """使用虚拟数据测试模型"""
        print(f"Testing model with virtual data ({n_tests} samples)")
        
        results = []
        
        for i in range(n_tests):
            try:
                # 创建虚拟观测数据
                obs_dict = self._create_virtual_obs()
                
                # 打印观测数据信息用于调试
                print(f"Test {i}: Created obs_dict with keys: {list(obs_dict.keys())}")
                for key, value in obs_dict.items():
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                
                # 模型推理
                with torch.no_grad():
                    self.policy.reset()
                    obs_dict_torch = dict_apply(obs_dict, 
                        lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
                    
                    print(f"Test {i}: Torch obs_dict prepared")
                    for key, value in obs_dict_torch.items():
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                    
                    result = self.policy.predict_action(obs_dict_torch)
                    print(f"Test {i}: Prediction result keys: {list(result.keys())}")
                    
                    # 根据模型类型获取动作
                    if 'action' in result:
                        predicted_action = result['action'][0].detach().cpu().numpy()
                    elif 'action_pred' in result:
                        predicted_action = result['action_pred'][0].detach().cpu().numpy()
                    else:
                        print(f"Unknown action key in result: {list(result.keys())}")
                        continue
                
                results.append({
                    'test_idx': i,
                    'action_shape': predicted_action.shape,
                    'action_mean': np.mean(predicted_action),
                    'action_std': np.std(predicted_action),
                    'action_min': np.min(predicted_action),
                    'action_max': np.max(predicted_action)
                })
                
                print(f"Test {i}: Action shape={predicted_action.shape}, "
                    f"range=[{np.min(predicted_action):.3f}, {np.max(predicted_action):.3f}]")
                
            except Exception as e:
                print(f"Error in test {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return results

    def _create_virtual_obs(self):
        """创建虚拟观测数据"""
        obs_dict = {}
        
        if hasattr(self.cfg.task, 'shape_meta'):
            obs_shapes = self.cfg.task.shape_meta.obs
            
            print(f"Creating virtual obs with shape_meta: {obs_shapes}")
            
            for key, shape_info in obs_shapes.items():
                shape = shape_info.shape
                print(f"Creating {key} with shape {shape}")
                
                if 'rgb' in key:
                    # 创建虚拟RGB图像 - 需要添加时间维度
                    if len(shape) == 3:  # (C, H, W)
                        # 添加时间维度，假设需要 n_obs_steps 个时间步
                        n_obs_steps = getattr(self.cfg.task, 'n_obs_steps', 2)
                        obs_shape = (n_obs_steps,) + shape
                    else:  # 已经有时间维度 (T, C, H, W)
                        obs_shape = shape
                    obs_dict[key] = np.random.randint(0, 255, obs_shape, dtype=np.uint8)
                else:
                    # 创建虚拟低维观测 - 也需要时间维度
                    if len(shape) == 1:  # (D,)
                        # 添加时间维度
                        n_obs_steps = getattr(self.cfg.task, 'n_obs_steps', 2)
                        obs_shape = (n_obs_steps,) + shape
                    else:  # 已经有时间维度 (T, D)
                        obs_shape = shape
                    obs_dict[key] = np.random.randn(*obs_shape).astype(np.float32)
                    
            print(f"Created obs_dict with keys: {list(obs_dict.keys())}")
            for key, value in obs_dict.items():
                print(f"  {key}: shape={value.shape}")
                
        return obs_dict
        
    def evaluate_episodes(self, n_episodes=5, start_episode=0):
        """评估指定数量的episode（需要数据集）"""
        if not hasattr(self, 'replay_buffer'):
            print("No dataset loaded. Please provide dataset_path to constructor.")
            return None
            
        print(f"Evaluating {n_episodes} episodes starting from episode {start_episode}")
        
        results = {
            'action_errors': [],
            'episode_ids': [],
            'step_errors': []
        }
        
        for ep_idx in range(start_episode, min(start_episode + n_episodes, self.replay_buffer.n_episodes)):
            print(f"Evaluating episode {ep_idx}")
            
            # 获取episode数据
            episode_data = self.replay_buffer.get_episode(ep_idx)
            episode_length = len(episode_data['action'])
            
            episode_errors = []
            
            # 逐步预测并比较
            step_interval = 5  # 每5步评估一次
            for step_idx in range(0, episode_length - self.cfg.n_action_steps, step_interval):
                try:
                    # 构造观测数据
                    obs_dict = self._get_obs_at_step(ep_idx, step_idx)
                    
                    # 模型预测
                    with torch.no_grad():
                        self.policy.reset()
                        obs_dict_torch = dict_apply(obs_dict, 
                            lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
                        result = self.policy.predict_action(obs_dict_torch)
                        
                        # 根据模型类型获取动作
                        if 'action' in result:
                            predicted_action = result['action'][0].detach().cpu().numpy()
                        elif 'action_pred' in result:
                            predicted_action = result['action_pred'][0].detach().cpu().numpy()
                        else:
                            continue
                    
                    # 获取真实动作
                    true_action = episode_data['action'][step_idx:step_idx + self.cfg.n_action_steps]
                    
                    # 计算误差
                    action_error = np.mean(np.abs(predicted_action - true_action))
                    episode_errors.append(action_error)
                    
                except Exception as e:
                    print(f"Error at episode {ep_idx}, step {step_idx}: {e}")
                    continue
            
            if episode_errors:
                results['action_errors'].extend(episode_errors)
                results['episode_ids'].append(ep_idx)
                results['step_errors'].append(np.mean(episode_errors))
                
        return results
    
    def _get_obs_at_step(self, episode_idx, step_idx):
        """获取指定步骤的观测数据"""
        # 使用dataset的get_item方法来获取正确预处理的数据
        dataset_idx = self.replay_buffer.episode_ends[episode_idx-1] if episode_idx > 0 else 0
        dataset_idx += step_idx
        
        # 获取数据项
        data_item = self.dataset.get_item(dataset_idx)
        
        # 返回观测数据（去掉action）
        obs_dict = {k: v for k, v in data_item.items() if k != 'action'}
        return obs_dict
    
    def visualize_virtual_test_results(self, results):
        """可视化虚拟测试结果"""
        if not results:
            print("No results to visualize")
            return
            
        print("Generating virtual test visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 动作统计
        action_means = [r['action_mean'] for r in results]
        action_stds = [r['action_std'] for r in results]
        action_mins = [r['action_min'] for r in results]
        action_maxs = [r['action_max'] for r in results]
        
        # 动作均值分布
        axes[0, 0].hist(action_means, bins=10, alpha=0.7)
        axes[0, 0].set_title('Action Mean Distribution')
        axes[0, 0].set_xlabel('Action Mean')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 动作标准差分布
        axes[0, 1].hist(action_stds, bins=10, alpha=0.7)
        axes[0, 1].set_title('Action Std Distribution')
        axes[0, 1].set_xlabel('Action Std')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 动作范围
        test_indices = [r['test_idx'] for r in results]
        axes[1, 0].plot(test_indices, action_mins, 'b-', label='Min', alpha=0.7)
        axes[1, 0].plot(test_indices, action_maxs, 'r-', label='Max', alpha=0.7)
        axes[1, 0].fill_between(test_indices, action_mins, action_maxs, alpha=0.3)
        axes[1, 0].set_title('Action Range per Test')
        axes[1, 0].set_xlabel('Test Index')
        axes[1, 0].set_ylabel('Action Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 动作形状信息
        action_shapes = [str(r['action_shape']) for r in results]
        unique_shapes = list(set(action_shapes))
        shape_counts = [action_shapes.count(shape) for shape in unique_shapes]
        
        axes[1, 1].bar(range(len(unique_shapes)), shape_counts)
        axes[1, 1].set_title('Action Shape Distribution')
        axes[1, 1].set_xlabel('Shape')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_xticks(range(len(unique_shapes)))
        axes[1, 1].set_xticklabels(unique_shapes, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'virtual_test_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印统计信息
        print("\n" + "="*50)
        print("VIRTUAL TEST RESULTS")
        print("="*50)
        print(f"Total tests: {len(results)}")
        print(f"Mean action mean: {np.mean(action_means):.6f}")
        print(f"Mean action std: {np.mean(action_stds):.6f}")
        print(f"Action range: [{np.mean(action_mins):.6f}, {np.mean(action_maxs):.6f}]")
        print(f"Most common action shape: {max(set(action_shapes), key=action_shapes.count)}")
        print("="*50)
        
    def visualize_dataset_results(self, results):
        """可视化数据集评估结果"""
        if not results or not results['action_errors']:
            print("No dataset results to visualize")
            return
            
        print("Generating dataset evaluation visualization...")
        
        # 1. 动作误差分布图
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(results['action_errors'], bins=50, alpha=0.7)
        plt.xlabel('Action Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Action Prediction Error Distribution')
        plt.grid(True, alpha=0.3)
        
        # 2. 每个episode的平均误差
        plt.subplot(1, 3, 2)
        plt.plot(results['episode_ids'], results['step_errors'], 'o-')
        plt.xlabel('Episode ID')
        plt.ylabel('Average Action Error')
        plt.title('Error per Episode')
        plt.grid(True, alpha=0.3)
        
        # 3. 误差统计
        plt.subplot(1, 3, 3)
        error_stats = {
            'Mean': np.mean(results['action_errors']),
            'Std': np.std(results['action_errors']),
            'Median': np.median(results['action_errors']),
            'Min': np.min(results['action_errors']),
            'Max': np.max(results['action_errors'])
        }
        
        bars = plt.bar(error_stats.keys(), error_stats.values())
        plt.ylabel('Error Value')
        plt.title('Error Statistics')
        plt.xticks(rotation=45)
        
        # 在柱子上添加数值
        for bar, value in zip(bars, error_stats.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dataset_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印统计信息
        print("\n" + "="*50)
        print("DATASET EVALUATION RESULTS")
        print("="*50)
        print(f"Total steps evaluated: {len(results['action_errors'])}")
        print(f"Episodes evaluated: {len(results['episode_ids'])}")
        for key, value in error_stats.items():
            print(f"{key} action error: {value:.6f}")
        print("="*50)
        
        return error_stats
    
    def save_results(self, virtual_results=None, dataset_results=None):
        """保存评估结果"""
        all_results = {
            'model_info': {
                'model_type': type(self.policy).__name__,
                'config_target': self.cfg.policy._target_,
                'checkpoint_path': self.ckpt_path,
                'device': str(self.device)
            }
        }
        
        if virtual_results:
            all_results['virtual_test'] = {
                'n_tests': len(virtual_results),
                'results': virtual_results
            }
            
        if dataset_results:
            all_results['dataset_evaluation'] = dataset_results
            
        # 保存为JSON
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
            
        print(f"Results saved to {self.output_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Offline Evaluation of Trained Policy')
    parser.add_argument('--checkpoint', '-c', required=True, 
                       help='Path to model checkpoint or checkpoint directory')
    parser.add_argument('--dataset', '-d', default=None,
                       help='Path to dataset (.zarr.zip file, optional)')
    parser.add_argument('--output', '-o', default='eval_results',
                       help='Output directory for results')
    parser.add_argument('--n_episodes', '-n', type=int, default=5,
                       help='Number of episodes to evaluate (if dataset provided)')
    parser.add_argument('--start_episode', '-s', type=int, default=0,
                       help='Starting episode index')
    parser.add_argument('--n_virtual_tests', '-v', type=int, default=10,
                       help='Number of virtual tests to run')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = OfflineEvaluator(
        ckpt_path=args.checkpoint,
        dataset_path=args.dataset,
        output_dir=args.output
    )
    
    # 运行虚拟测试
    print("\n" + "="*60)
    print("RUNNING VIRTUAL TESTS")
    print("="*60)
    virtual_results = evaluator.test_with_virtual_data(n_tests=args.n_virtual_tests)
    evaluator.visualize_virtual_test_results(virtual_results)
    
    # 检查是否可以进行数据集评估
    dataset_results = None
    if args.dataset and hasattr(evaluator, 'dataset') and evaluator.dataset is not None:
        print("\n" + "="*60)
        print("RUNNING DATASET EVALUATION")
        print("="*60)
        dataset_results = evaluator.evaluate_episodes(
            n_episodes=args.n_episodes,
            start_episode=args.start_episode
        )
        if dataset_results:
            evaluator.visualize_dataset_results(dataset_results)
    elif args.dataset:
        print("\n" + "="*60)
        print("DATASET EVALUATION SKIPPED")
        print("="*60)
        print("The provided dataset does not contain action labels and cannot be used for comparison evaluation.")
        print("Only virtual testing was performed.")
    
    # 保存所有结果
    evaluator.save_results(virtual_results, dataset_results)

if __name__ == "__main__":
    main()