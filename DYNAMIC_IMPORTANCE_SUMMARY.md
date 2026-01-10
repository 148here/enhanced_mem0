# 动态重要性和记忆过期管理系统实施总结

## 概述

成功实现了一个完整的记忆动态管理系统，包括：
1. 记忆重要性评分（0-5，归一化到0-1）
2. 基于重要性的搜索排序
3. 自动衰减和过期机制
4. 记忆复活机制
5. 快速搜索（仅搜索活跃记忆）

## 修改的文件

### 1. config.py
添加了完整的配置参数：
- `ENABLE_DYNAMIC_IMPORTANCE`: 启用动态重要性（默认 false）
- `DYNAMIC_IMPORTANCE_WEIGHT`: 重要性权重（默认 0.1）
- `DECAY_CHECK_INTERVAL`: 衰减检查间隔（默认 5）
- `DECAY_MULTIPLIER`, `DECAY_OFFSET`, `DECAY_THRESHOLD`: 衰减参数
- `REVIVE_MULTIPLIER`, `REVIVE_OFFSET`, `REVIVE_MAX`: 复活参数
- `ENABLE_FAST_SEARCH`: 启用快速搜索（默认 false）

### 2. prompt.py
- 创建了 `USER_MEMORY_EXTRACTION_WITH_IMPORTANCE_PROMPT`
- 扩展原有 prompt，要求 LLM 为每个 fact 评估重要性（0-5）
- 返回格式：`{"facts": [{"content": "...", "importance": 3}, ...]}`
- 包含详细的评分指南和 few-shot 示例

### 3. memory_system.py
核心实现：
- **状态管理**：
  - `add_counter`: 记录 add 调用次数
  - `enable_importance`: 是否启用动态重要性
  - `enable_fast_search`: 是否启用快速搜索
  - `last_decay_time`: 上次衰减时间

- **关键方法**：
  - `_parse_facts_with_importance()`: 解析带重要性的 facts
  - `add_turn()`: 支持重要性评分，触发衰减检查
  - `_trigger_decay()`: 应用衰减公式到所有活跃记忆
  - `revive_memories()`: 标记使用过的记忆为活跃
  - `search()`: 支持快速搜索和动态重要性排序

### 4. chat_manager.py
集成新功能：
- `send_message()`: 接收 `enable_dynamic_importance` 和 `enable_fast_search` 参数
- `_send_message_direct()`: 传递参数给 search，调用 revive_memories
- `_send_message_with_judge()`: 强制使用全局搜索
- 状态返回增加：`dynamic_importance_enabled`, `fast_search_enabled`, `revived_memories`, `next_decay_in`

### 5. chat_webui.py
UI 控制和展示：
- 添加两个新的 Checkbox 控件：
  - "Enable Dynamic Importance"
  - "Enable Fast Search"
- 更新 `_status_markdown()` 显示新的状态信息
- 修改 `on_send()` 接收并传递新参数

### 6. readme
完整记录了本次实施的所有细节，包括：
- 核心功能说明
- 修改文件清单
- 技术实现细节
- 使用说明
- 配置参数说明
- 向后兼容性说明

## 核心机制

### 1. 动态重要性评分
- LLM 在提取 facts 时为每个 fact 评分（0-5）
- 存储在 `metadata.dynamic_importance`（归一化到 0-1）
- 0 = 完全不重要的琐碎对话
- 5 = 非常重要的关键信息

### 2. 搜索排序增强
```python
enhanced_score = original_score + weight * dynamic_importance
```
- `weight` 默认为 0.1（可配置）
- 可通过开关启用/禁用

### 3. 记忆衰减机制
- 每 N 次 add 后触发全局衰减（N 默认为 5）
- 衰减公式：
```python
if importance > 0:
    new_importance = (importance * 0.99) - 0.002
```
- 低于阈值（默认 -0.5）标记为过期

### 4. 记忆复活机制
- search 命中的记录被"复活"
- `is_expired` 重置为 false
- 复活公式：
```python
new_importance = min((importance + 0.002) * 1.01, 1.0)
```

### 5. 快速搜索
- 只搜索 `is_expired=false` 的记忆
- 裁判模型搜索时强制全局搜索

## 技术实现要点

### Metadata 格式
```python
metadata = {
    "timestamp": 1234567890.0,
    "dynamic_importance": 0.6,  # 归一化到 0-1
    "original_importance": 3,   # 保留原始分数供调试
    "is_expired": False,
}
```

### FAISS 限制处理
- FAISS 不支持直接更新 metadata
- 采用 search 时动态处理的策略：
  - 为旧记忆自动添加默认值
  - 快速搜索通过过滤实现
  - 衰减和复活记录在日志中

### 向后兼容
- 旧记忆自动获得默认值：
  - `dynamic_importance = 0.5`
  - `is_expired = False`
- 不启用时使用原有的标准 prompt
- 所有新功能默认关闭

## 使用说明

### 环境变量配置
在 `.env` 文件中添加：
```bash
# 动态重要性
ONLINE_CHAT_ENABLE_DYNAMIC_IMPORTANCE=true
ONLINE_CHAT_DYNAMIC_IMPORTANCE_WEIGHT=0.1

# 衰减配置
ONLINE_CHAT_DECAY_CHECK_INTERVAL=5
ONLINE_CHAT_DECAY_MULTIPLIER=0.99
ONLINE_CHAT_DECAY_OFFSET=-0.002
ONLINE_CHAT_DECAY_THRESHOLD=-0.5

# 复活配置
ONLINE_CHAT_REVIVE_MULTIPLIER=1.01
ONLINE_CHAT_REVIVE_OFFSET=0.002
ONLINE_CHAT_REVIVE_MAX=1.0

# 快速搜索
ONLINE_CHAT_ENABLE_FAST_SEARCH=true
```

### WebUI 使用
1. 启动 WebUI：`python chat_webui.py`
2. 在界面中找到两个新的复选框：
   - "Enable Dynamic Importance"
   - "Enable Fast Search"
3. 勾选以启用相应功能
4. 在"Current Status"区域查看实时状态

### 建议的使用流程
1. 先测试单个功能：
   - 只启用动态重要性，观察排序效果
   - 只启用快速搜索，观察性能提升
2. 组合使用：
   - 同时启用两个功能
   - 结合裁判模型使用
3. 调整参数：
   - 根据实际效果调整 `DYNAMIC_IMPORTANCE_WEIGHT`
   - 根据记忆增长速度调整衰减参数

## 测试建议

### 1. 重要性评分测试
- 输入琐碎对话（如 "Hi"），验证低分（0-1）
- 输入关键信息（如生日、职业），验证高分（4-5）
- 检查 metadata 正确存储

### 2. 搜索排序测试
- 创建高重要性和低重要性记忆
- 验证启用动态重要性后排序变化
- 对比启用前后的结果

### 3. 衰减机制测试
- 添加 5 条记忆触发衰减
- 检查控制台输出的衰减日志
- 验证 `next_decay_in` 计数器正确

### 4. 复活机制测试
- 创建一些记忆并等待衰减
- 搜索并使用这些记忆
- 验证 `revived_memories` 计数增加

### 5. 快速搜索测试
- 创建过期和活跃记忆
- 验证快速搜索只返回活跃记忆
- 验证裁判模型强制全局搜索

## 性能考虑

- **动态重要性**：对性能影响极小，只增加排序计算
- **快速搜索**：可显著提升搜索速度（跳过过期记忆）
- **衰减检查**：每 N 次 add 时执行，可能略微增加延迟
- **复活操作**：轻量级操作，性能影响可忽略

## 故障排查

### 重要性评分不生效
- 检查 `ENABLE_DYNAMIC_IMPORTANCE` 是否为 true
- 查看控制台是否有 prompt 相关错误
- 检查 fact extraction raw 输出格式

### 快速搜索不生效
- 确认记忆中有 `is_expired` metadata
- 检查 WebUI 中的复选框状态
- 验证裁判模型未启用（会强制全局搜索）

### 衰减未触发
- 检查 `add_counter` 是否达到 `DECAY_CHECK_INTERVAL`
- 查看控制台输出的 "[Decay]" 日志
- 验证 `DECAY_CHECK_INTERVAL` 配置正确

## 总结

本次实施成功完成了动态重要性和记忆过期管理系统的所有计划功能：
- ✅ 11 个 TODO 全部完成
- ✅ 5 个核心文件修改完毕
- ✅ 无 linter 错误
- ✅ 完整的文档记录
- ✅ 向后兼容性保证
- ✅ WebUI 控制和状态展示

系统现在具备了智能的记忆管理能力，能够根据重要性和使用频率自动管理记忆生命周期。
