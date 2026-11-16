# 本文件是 Prompt 模板类的顶层再导出模块，只做一件事：把 trainer.utils 里的 Prompt 暴露在仓库根路径下
# 作用是兼容旧版导入路径：sft.py、DynamicPO.py、inference.py 都写从 Prompt 导入 Prompt，无需改写为 trainer.utils
# 这样训练、推理、评估各入口共享同一个 Prompt 类，统一模板渲染逻辑
# 从 trainer.utils 导入真正的 Prompt 类，本文件不重新实现任何逻辑
from trainer.utils import Prompt

# 用 __all__ 声明对外只导出 Prompt，使通配导入时只暴露该类、保持命名空间干净
__all__ = ["Prompt"]
