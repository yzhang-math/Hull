import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xgboost as xgb
from typing import Optional, Tuple
import math

"""
简化版S&P 500预测模型 - 仅特征级Transformer

设计理念：
- 输入：(batch_size, num_features) - 每个样本是一组技术指标
- 每个特征作为一个token
- 通过Attention学习特征之间的交互关系
- 不使用位置编码（特征无位置关系）
- 最适合使用技术指标（RSI、MACD等）进行预测
"""


class FeatureAttention(nn.Module):
    """
    特征级多头注意力机制
    学习特征之间的交互关系
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_features, d_model)
        Returns:
            output: (batch_size, num_features, d_model)
        """
        batch_size, num_features, _ = x.shape
        
        # 投影到Q, K, V
        Q = self.w_q(x).view(batch_size, num_features, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, num_features, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, num_features, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用attention
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_features, self.d_model)
        
        output = self.w_o(context)
        return output


class FeatureTransformerLayer(nn.Module):
    """
    特征级Transformer层
    包含self-attention和feed-forward网络
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = FeatureAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class FeatureLevelTransformer(nn.Module):
    """
    特征级Transformer模型
    
    输入：(batch_size, num_features)
    处理：将每个特征映射到高维空间，通过Transformer学习特征交互
    输出：(batch_size, output_dim)
    
    不使用位置编码！
    """
    def __init__(
        self,
        feature_dim: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        output_dim: Optional[int] = 32
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.d_model = d_model
        
        # 将每个标量特征映射到d_model维度
        self.feature_embedding = nn.Linear(1, d_model)
        
        # Transformer层
        self.layers = nn.ModuleList([
            FeatureTransformerLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出投影
        if output_dim is not None:
            self.output_projection = nn.Sequential(
                nn.Linear(feature_dim * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, output_dim)
            )
            self.output_dim = output_dim
        else:
            self.output_projection = nn.Identity()
            self.output_dim = feature_dim * d_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, feature_dim) - 原始特征
        Returns:
            output: (batch_size, output_dim)
        """
        batch_size = x.shape[0]
        
        # 将每个特征映射到d_model维度
        # (batch_size, feature_dim) -> (batch_size, feature_dim, 1) -> (batch_size, feature_dim, d_model)
        x = x.unsqueeze(-1)  # (batch_size, feature_dim, 1)
        x = self.feature_embedding(x)  # (batch_size, feature_dim, d_model)
        
        # 通过Transformer层（在特征维度上做attention）
        for layer in self.layers:
            x = layer(x)
        
        # Flatten并投影到输出维度
        x = x.reshape(batch_size, -1)  # (batch_size, feature_dim * d_model)
        output = self.output_projection(x)  # (batch_size, output_dim)
        
        return output


class SP500Predictor:
    """
    S&P 500预测模型
    结合特征级Transformer和XGBoost
    """
    def __init__(
        self,
        feature_dim: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        output_dim: Optional[int] = 32,
        xgb_params: Optional[dict] = None,
        combine_features: bool = True,
        # -----------------
        # MODIFICATION: Added batch size for inference/feature extraction
        transformer_batch_size: int = 128
        # -----------------
    ):
        """
        Args:
            feature_dim: 输入特征数量
            d_model: 特征嵌入维度
            num_heads: attention头数
            num_layers: Transformer层数
            d_ff: feed-forward维度
            dropout: dropout比例
            output_dim: Transformer输出维度
            xgb_params: XGBoost参数
            combine_features: 是否组合原始特征和Transformer特征
            transformer_batch_size: 用于feature extraction的batch size
        """
        self.transformer = FeatureLevelTransformer(
            feature_dim=feature_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            output_dim=output_dim
        )
        
        self.combine_features = combine_features
        # -----------------
        # MODIFICATION: Store the batch size
        self.transformer_batch_size = transformer_batch_size
        # -----------------
        
        # XGBoost参数（针对金融数据优化）
        default_xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_estimators': 100
        }
        
        if xgb_params is not None:
            default_xgb_params.update(xgb_params)
        self.xgb_params = default_xgb_params
        
        self.xgboost_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transformer.to(self.device)

    # -----------------------------------------------------------------
    # MODIFICATION: extract_features now runs in batches
    # -----------------------------------------------------------------    
    def extract_features(self, X: torch.Tensor, batch_size: Optional[int] = None) -> np.ndarray:
        """
        使用Transformer提取特征 (in batches to prevent OOM)
        
        Args:
            X: (batch_size, feature_dim)
            batch_size: The batch size to use. If None, uses self.transformer_batch_size
        Returns:
            features: (batch_size, output_dim)
        """
        self.transformer.eval()
        
        b_size = batch_size if batch_size is not None else self.transformer_batch_size
        all_features = []
        
        with torch.no_grad():
            for i in range(0, X.shape[0], b_size):
                end_idx = min(i + b_size, X.shape[0])
                X_batch = X[i:end_idx].to(self.device)
                
                features_batch = self.transformer(X_batch)
                all_features.append(features_batch.cpu().numpy())
                
        return np.concatenate(all_features, axis=0)
    # -----------------------------------------------------------------
    # END OF MODIFICATION
    # -----------------------------------------------------------------
    
    def fit_transformer(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor, 
        epochs: int = 50,
        batch_size: int = 64, # This is for TRAINING the transformer
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ):
        """
        训练Transformer特征提取器
        """
        self.transformer.train()
        optimizer = torch.optim.AdamW(
            self.transformer.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        X = X.to(self.device)
        y = y.to(self.device)
        
        # 划分训练和验证集（时间序列：不shuffle）
        n_samples = X.shape[0]
        n_val = int(n_samples * validation_split)
        
        X_train, y_train = X[:-n_val], y[:-n_val]
        X_val, y_val = X[-n_val:], y[-n_val:]
        
        # 简单的预测头用于预训练
        pred_head = nn.Linear(self.transformer.output_dim, 1).to(self.device)
        optimizer_head = torch.optim.AdamW(pred_head.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.transformer.train()
            pred_head.train()
            
            # Mini-batch训练
            n_batches = (len(X_train) + batch_size - 1) // batch_size
            train_loss = 0
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_train))
                
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                optimizer.zero_grad()
                optimizer_head.zero_grad()
                
                features = self.transformer(X_batch)
                predictions = pred_head(features).squeeze()
                
                loss = F.mse_loss(predictions, y_batch)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(pred_head.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer_head.step()
                
                train_loss += loss.item()
            
            train_loss /= n_batches
            
            # 验证 (using batched extraction for validation)
            self.transformer.eval()
            pred_head.eval()
            with torch.no_grad():
                # Use extract_features to handle potentially large val set
                val_features_np = self.extract_features(X_val)
                val_features = torch.tensor(val_features_np).to(self.device)
                
                val_predictions = pred_head(val_features).squeeze()
                val_loss = F.mse_loss(val_predictions, y_val).item()
            
            if verbose and epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f'Early stopping at epoch {epoch}')
                    break
    
    def fit_xgboost(self, X: torch.Tensor, y: np.ndarray, eval_set: Optional[Tuple] = None):
        """
        训练XGBoost模型
        """
        # -----------------
        # MODIFICATION: This call now uses the batched extract_features
        transformer_features = self.extract_features(X)
        # -----------------
        
        # 组合特征
        if self.combine_features:
            X_combined = np.concatenate([
                X.cpu().numpy(),
                transformer_features
            ], axis=1)
        else:
            X_combined = transformer_features
        
        # 准备验证集
        eval_sets = None
        if eval_set is not None:
            X_val, y_val = eval_set
            
            # -----------------
            # MODIFICATION: This call also uses the batched extract_features
            transformer_features_val = self.extract_features(X_val)
            # -----------------
            
            if self.combine_features:
                X_val_combined = np.concatenate([
                    X_val.cpu().numpy(),
                    transformer_features_val
                ], axis=1)
            else:
                X_val_combined = transformer_features_val
            
            eval_sets = [(X_combined, y), (X_val_combined, y_val)]
        
        # 训练XGBoost
        self.xgboost_model = xgb.XGBRegressor(**self.xgb_params)
        
        if eval_sets is not None:
            self.xgboost_model.fit(X_combined, y, eval_set=eval_sets, verbose=False)
        else:
            self.xgboost_model.fit(X_combined, y)
    
    def fit(
        self,
        X: torch.Tensor,
        y: np.ndarray,
        val_X: Optional[torch.Tensor] = None,
        val_y: Optional[np.ndarray] = None,
        transformer_epochs: int = 50,
        transformer_lr: float = 1e-3,
        verbose: bool = True
    ):
        """
        端到端训练：Transformer + XGBoost
        """
        if verbose:
            print("Step 1: Training Transformer feature extractor...")
        
        y_tensor = torch.FloatTensor(y)
        self.fit_transformer(
            X, y_tensor,
            epochs=transformer_epochs,
            lr=transformer_lr,
            verbose=verbose
        )
        
        if verbose:
            print("\nStep 2: Training XGBoost on extracted features...")
        
        eval_set = (val_X, val_y) if val_X is not None and val_y is not None else None
        
        # This will now use the batched feature extraction
        self.fit_xgboost(X, y, eval_set=eval_set)
        
        if verbose:
            print("Training complete!")
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        预测
        """
        # -----------------
        # MODIFICATION: This call now uses the batched extract_features
        transformer_features = self.extract_features(X)
        # -----------------
        
        if self.combine_features:
            X_combined = np.concatenate([
                X.cpu().numpy(),
                transformer_features
            ], axis=1)
        else:
            X_combined = transformer_features
        
        predictions = self.xgboost_model.predict(X_combined)
        return predictions
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'transformer_state_dict': self.transformer.state_dict(),
            'xgboost_model': self.xgboost_model,
            'combine_features': self.combine_features
        }, path)
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.transformer.load_state_dict(checkpoint['transformer_state_dict'])
        self.xgboost_model = checkpoint['xgboost_model']
        self.combine_features = checkpoint['combine_features']