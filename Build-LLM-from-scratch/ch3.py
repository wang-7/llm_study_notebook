# %% [markdown]
# ## 3.4 用一个类实现自注意力

# %%
import torch.nn as nn
import torch

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    
    def forward(self, x):
        # x shape: [seq_len, embed_dim]
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        # attn score: [seq_len, seq_len], 表示每个token和每个token的相关性分数
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values

        return context_vec


# %% [markdown]
# 注意到上面的qkv参数矩阵是用nn.Parameter手动赋值的，我们也可以用nn.Linear来实现更好的初始化策略。

# %%
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        # x shape: [seq_len, embed_dim]
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # attn score: [seq_len, seq_len], 表示每个token和每个token的相关性分数
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values

        return context_vec

# %%
# 下面构造一个输入数据来进行测试
seq_len = 6
embed_dim = 3
torch.manual_seed(42)
inputs = torch.tensor( [[0.43, 0.15, 0.89], # Your (x^1) 
                        [0.55, 0.87, 0.66], # journey (x^2) 
                        [0.57, 0.85, 0.64], # starts (x^3) 
                        [0.22, 0.58, 0.33], # with (x^4) 
                        [0.77, 0.25, 0.10], # one (x^5) 
                        [0.05, 0.80, 0.55]] # step (x^6) 
                        )
print(inputs)

d_out = 4
sa_v1 = SelfAttention_v1(embed_dim, d_out)
sa_v2 = SelfAttention_v2(embed_dim, d_out)

context_vec1 = sa_v1(inputs)
context_vec2= sa_v2(inputs)
print('context vector v1:\n', context_vec1)
print('context_vector v2:\n ', context_vec2)

# %% [markdown]
# 可以观察到，v1和v2因为参数矩阵的初始化不同，导致输出不同，我们可以将v2的参数矩阵复制到v1上去，使其输出相同。

# %%
# print(sa_v1.state_dict())
# print(sa_v2.state_dict())
#
sa_v1.W_key.data.copy_(sa_v2.W_key.weight.data.T)
sa_v1.W_query.data.copy_(sa_v2.W_query.weight.data.T)
sa_v1.W_value.data.copy_(sa_v2.W_value.weight.data.T)

context_vec1 = sa_v1(inputs)
context_vec2= sa_v2(inputs)
print('context vector v1:\n', context_vec1)
print('context_vector v2:\n ', context_vec2)

# %% [markdown]
# ## 3.5 用因果注意力隐藏未来的单词

# %%
# 首先借用上面写的注意力参数矩阵计算注意力权重
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_score = queries @ keys.T
attn_weights = torch.softmax(attn_score / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

# %%
context_length = keys.shape[0]
# 使用tril函数生成一个矩阵的下三角矩阵（其余元素置零）
mask_simple = torch.tril(attn_weights)
print(mask_simple)

# %% [markdown]
# 注意到应用tril后，剩下元素不再满足和为1，因此需要重新归一化。

# %%
attn_weights_renorm = mask_simple / torch.sum(mask_simple, dim=-1, keepdim=True)
print(attn_weights_renorm)


# %% [markdown]
# 此处需要注意，实际上进行了两次归一化操作，那能否简化呢？观察下面的例子

# %%
mask = torch.triu(torch.ones_like(attn_score), diagonal=1)
masked = attn_score.masked_fill(mask.bool(), -torch.inf)
print(masked)

# %%
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

# %% [markdown]
# 发现这里一步归一化算得的注意力权重和上面的两步归一化算得的权重是一模一样的。这是因为上面两步归一化在数学上等价于在未被掩码的子集上进行归一化，这一结论可以简单地通过定义证明。
# 
# 下面还要实现dropout

# %%
# 构造一个全1的测试矩阵
torch.manual_seed(42)
example = torch.ones(6, 6)
dropout = torch.nn.Dropout(0.5)
print(dropout(example))

# %% [markdown]
# 注意到dropout的作用是根据概率随机地将一部分参数置零，同时为了保持数据规模，其余元素除以(1-p)，相当于保持期望不变。通常在self-attention中应用dropout有两个地方，attention weights或者values，我们采用前者。
# 
# Dropout在训练阶段每次前向传播时都独立地起作用，测试阶段不起作用。

# %%
# 现在可以将dropout应用于注意力权重上了。
torch.manual_seed(42)
print(dropout(attn_weights))

# %% [markdown]
# 最后，我们将上面两种操作（mask，dropout）添加到attention类中，同时让其能够处理多批次输入。

# %%
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', 
                             torch.triu(torch.ones(context_length, context_length), diagonal=1))
        
    def forward(self, x):
        # x.shape=[batch_size, context_length, d_in]
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 注意此处有batch维度的存在，不能直接转置（.T运算符会反转所有维度）
        attn_scores = queries @ torch.transpose(keys, 1, 2)
        # 先对score应用softmax得到权重
        attn_scores.masked_fill_(self.mask.bool(), -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1) 
        # 再对权重应用dropout
        attn_weights = self.dropout(attn_weights)
        # 最后计算value
        context_vec = attn_weights @ values

        return context_vec        



# %%
print(inputs.shape)
batch_inputs = torch.stack([inputs, inputs], dim=0)
print(batch_inputs.shape)
batch_size, context_len, embed_dim = batch_inputs.shape
ca = CausalAttention(embed_dim, 2, context_len, 0.0)
context_vec = ca(batch_inputs)
print(context_vec.shape)

# %% [markdown]
# ## 3.6 将单头注意力扩展为多头注意力

# %% [markdown]
# 多头注意力的实现，简单来说就是多个单头注意力（上面实现的）并行计算，将结果拼接即可。既然如此，一个简单的方法就是直接用列表包装单头注意力，但这样是用循环的串行计算，非常耗时。

# %%
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 使用nn.ModuleList自动注册子模块，否则pytorch不会自动注册列表内的模块。
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
                       for _ in range(num_heads)]
        )
    def forward(self, x):
        return torch.cat(
            [head(x) for head in self.heads], dim=-1
        )

# %% [markdown]
# 为了将上面的串行计算改为并行，需要用一些矩阵的拼接和分割操作。
# 
# 下面实现了高效的多头注意力机制，也是大模型中实际使用的多头注意力机制。

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_len, dropout, qkv_bias=False):
        super().__init__()

        #划分多头
        self.num_heads, self.d_out = num_heads, d_out
        assert d_out % num_heads == 0, 'd_out must be devisable by num_heads'
        self.d_head = d_out // num_heads

        # 首先初始化qkv矩阵，注意此处d_out实际上num_heads个长度为d_head的头拼接在一起的
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 注册mask, dropout层，projection层
        self.register_buffer('mask', torch.triu(torch.ones(context_len, context_len), diagonal=1))
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_out, d_out)

    def forward(self, x):
        # Step1: 计算qkv，并拆成多头堆叠的形式
        # x.shape = [batch_size, context_len, embed_dim]
        bs, context_len, embed_dim = x.shape
        queries = self.W_q(x).view(bs, context_len, self.num_heads, self.d_head)
        keys = self.W_k(x).view(bs, context_len, self.num_heads, self.d_head)
        values = self.W_v(x).view(bs, context_len, self.num_heads, self.d_head)
        queries, keys, values = queries.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)

        # Step2: 计算注意力权重，这一步和单头没有区别，注意softmax中有系数
        attn_scores = queries @ keys.transpose(2, 3) # [bs, num_head, context_len, context_len]
        # 对mask切片是必要的，因为输入的长度不一定刚好是初始化时设定的长度，应理解为最大长度
        attn_scores.masked_fill_(self.mask.bool()[:context_len, :context_len], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Step3: 计算上下文向量，并将多头拼接，注意view之前需要将tensor连续化。最后别忘记projection。
        context_vec = attn_weights @ values # [bs, num_head, context_len, d_head]
        context_vec = context_vec.transpose(1, 2).contiguous().view(bs, context_len, self.d_out)
        context_vec = self.projection(context_vec)

        return context_vec

# %%
bs, context_len, embed_dim = batch_inputs.shape
mha = MultiHeadAttention(embed_dim, 2, 2, context_len, 0.0)
context_vecs = mha(batch_inputs)
print(context_vec)
print(context_vec.shape)


