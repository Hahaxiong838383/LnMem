# LnMem Persona Assessment Tool

独立前端单页应用：人格画像测评 + 批量虚拟画像生成。

## 运行
1. 直接打开 `persona-assessment.html`。
2. 如浏览器限制本地 JSON 读取，建议使用本地静态服务（`python3 -m http.server`）。

## 交付内容
- `persona-assessment.html`: 主工具
- `data/*.json`: 5个量表题库
- `prompts/*.txt`: 分析与生成 prompt 模板
- `examples/`: 示例 JSON 与 HTML 报告

## 学术基础
- HEXACO (Lee & Ashton, 2004)
- ECR-R (Brennan, Clark & Shaver, 1998)
- GSE (Bandura)
- SDT (Deci & Ryan)
- Schwartz PVQ-10 (2012)
- CAPS 情境签名 (Mischel & Shoda)

## 说明
- 情境题共14题，覆盖压力、协作、亲密关系、能量管理、AI互动偏好。
- 分数由量表答案计算，LLM仅用于解释与扩展文字层。
- 数据仅存于 localStorage，无后端依赖。
