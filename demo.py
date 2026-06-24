import argparse
from config import Config
from uuid import uuid4
import os
from agent import *

def generate_question_info(task_type):
    base_info = {
        "machine_learning": {
            "question": "I have a dataset {heart_disease.csv}, the “Target” column is the target column and the other columns are the feature columns, please perform a singular value decomposition downscaling on this dataset.",
            "files": [{
                "name": "heart_disease.csv",
                "path": "data/heart_disease.csv"
            }]
        },
        "statistics_t_test": {
            "question": "There are two existing data tables, {data1.tsv} is a biomarker data table, with sample names as column names and biomarker names as row names, {group1.tsv} is a data table containing sample grouping information, where the two groups are 1 and 2, with the column name ID as sample names and the column name group as grouping information. Please use the tool t_test to perform an independent sample t-test based on this information.",
            "files": [
                {
                    "name": "data1.tsv", 
                    "path": "data/data1.tsv"
                },
                {
                    "name": "group1.tsv", 
                    "path": "data/group1.tsv"
                }
            ]
        },
        "statistics_qq_plot": {
            "question": "There is a dataset {boxplot.tsv}, Group2 column is the grouping information, which contains treat1 and treat2, please analyze whether the data in Value column of the two groupings of treat1 and treat2 have similar distribution characteristics, and plot QQ plot.",
            "files": [{
                "name": "boxplot.tsv", 
                "path": "data/boxplot.tsv"
            }]
        },
        "visualization_survival_plot": {
            "question": "There is a data table {TCGA_LIHC_survival.txt} that contains information related to patient survival. The column named OS.time represents the survival time of the patients, the column named OS indicates the survival event status of the patients, and the column named gender is a variable that affects survival. Please use the tool survival_curve to plot the survival curve based on this information.",
            "files": [{
                "name": "TCGA_LIHC_survival.txt",
                "path": "data/TCGA_LIHC_survival.txt"
            }]
        },
        "visualization_violin_plot": {
            "question": "There is a data file {plot.tsv}, the Sex column is the gender of the sample, Age_Group is the different age grouping corresponding to each gender, please use the violin plot to show the distribution of WBC for each age grouping of different genders.",
            "files": [{
                "name": "plot.tsv",
                "path": "data/plot.tsv"
            }]
        },
        "omics": {
            "question": "The following is the microarray expression data {GSM509787_E1507N.CEL.gz} of 1 sample from esophageal squamous cell carcinoma dataset GSE20347, the corresponding probe number is {GPL571}, please use the CeltoExp tool to convert the CEL microarray expression file into gene expression profile data in txt file format.",
            "files": [{
                "name": "GSM509787_E1507N.CEL.gz",
                "path": "data/GSM509787_E1507N.CEL.gz"
            }]
        },
        "test": {
            "question": "以下是食管鳞状细胞癌数据集GSE20347中1个样本的微阵列表达数据{GSM509787_E1507N.CEL.gz}，其对应的探针编号为{GPL571}，请使用CeltoExp工具将该CEL微阵列表达文件转换为txt文件格式的基因表达谱数据。",
            "files": [{
                "name": "GSM509787_E1507N.CEL.gz",
                "path": "data/GSM509787_E1507N.CEL.gz"
            }]
            
        },
        "fastqc": {
            "question": "以下是1个样本的fastq{GSM509787_E1507N.fastq}，请使用fastqc工具对其质量控制",
            "files": [{
                "name": "GSM509787_E1507N.fastq",
                "path": "data/GSM509787_E1507N.CEL.fastq"
            }]
            
        }
    }
    return base_info[task_type]


parser = argparse.ArgumentParser(description='Demo for BioMedAgent')
parser.add_argument('--task', type=str, choices=['machine_learning', 'statistics_t_test', 'statistics_qq_plot', 'visualization_survival_plot', 'visualization_violin_plot', 'omics', 'test', 'fastqc'],
                    default='machine_learning', help='task type')

args = parser.parse_args()
config = Config(f"{uuid4()}")

question_info = generate_question_info(args.task)

# 准备任务的时候会根据USE_FILE_APPENDIX是否为True判断是否要读取文件的部分内容，后续设计workflow的时候作为提示词
# task是多个智能体共享的对象
task = Task(question_info, config)

linguist = Linguist(task)
translator = Translator(task)
prompt_engineer = PromptEngineer(task)
file_analyst = FileAnalyst(task)
tool_scorer = ToolScorer(task)
tool_descriptor = ToolDescriptor(task)
tool_rescorer = ToolReScorer(task)
workflow_designer = WorkflowDesigner(task)
tool_analyst = ToolAnalyst(task)
workflow_formatter = WorkflowFormatter(task)
action_designer = ActionDesigner(task)
mermaid_designer = MermaidDesigner(task)
# 缺少programmer,写执行代码，写测试代码，把功能代码和测试代码合并成可执行代码，把功能代码和测试代码合并成可执行代码

summary_analyst = SummaryAnalyst(task)
workflow_redesigner = WorkflowReDesigner(task)

# 1. 查看输入是否是英文，输出yes or no
linguist.check_English_task()

# 2. 如果不是英文，翻译成英文
translator.translate_question()

# 3. 对问题做提示词增强，存储在status的refined_question字段，但是后文都没使用过
prompt_engineer.refine_prompt()

# 4. 根据翻译后的问题（question）和文件名称，详细描述数据文件
asyncio.run(file_analyst.analyse_files())

# 5. 根据问题、工具名称、使用文档，让LLM给出每个工具和问题的相关度，1-10分
asyncio.run(tool_scorer.tools_score())

# 6. 根据config文件中的 HIGHSCORE_TOOL_THRESHOLD，对于高于阈值的工具，选择让LLM进行详细的描述
asyncio.run(tool_descriptor.tools_describe())

# 7. 将选出来的高于阈值的这些工具的分数、描述作为上下文，进行二次打分（也许会选出新的工具）
asyncio.run(tool_rescorer.tools_score())

# 8. 对二次打分的结果（如果有新增的工具）生成详细的描述
asyncio.run(tool_descriptor.tools_describe())

# 9. 根据用户的问题、数据描述、高于WORKFLOW_USED_TOOL_THRESHOLD的工具描述、Redis的记忆文件，生成Goal、Tool、Input、Output、Description的多条结果的string字符串
workflow_designer.workflow_design()

# 10. 生成工具调用的建议
asyncio.run(tool_analyst.tools_analyse())

# 11. 将workflow文字描述，转换成按步骤执行的list
workflow_formatter.workflow_format()

# 12. 格式化成stage goal action tool input output的对象列表
asyncio.run(action_designer.actions_design())

# 13. 可视化workflow计划 
mermaid_designer.mermaid_design()

# 14. 初次编码、测试、执行、观察结果生成建议，如果不通过会重试3次，每次更新任务状态、重新设计工作流、重新设计工具调用、重新编码，超出次数不成功则失败

# ok, suggestion = programmer.programming()
# retry_times = 1
# while not ok and retry_times < 3:
#     retry_times += 1
#     task.backtrack_update()
#     workflow_redesigner.workflow_redesign(suggestion)
#     asyncio.run(tool_analyst.tools_analyse())
#     workflow_formatter.workflow_format()
#     asyncio.run(action_designer.actions_design())
#     task.status.status_update("workflow")
#     ok, suggestion = programmer.programming()
# if not ok: raise Exception("Task failed")


# 15. 总结执行结果
summary_analyst.summary()