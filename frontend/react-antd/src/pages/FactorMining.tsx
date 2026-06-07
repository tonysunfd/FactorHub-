import { useState, useEffect, useRef } from "react";
import {
  Card,
  Form,
  Input,
  DatePicker,
  Button,
  Select,
  InputNumber,
  Progress,
  Row,
  Col,
  message,
  Space,
  Divider,
  Tag,
  Spin,
  Alert,
  Tabs,
} from "antd";
import {
  PlayCircleOutlined,
  SaveOutlined,
  BarChartOutlined,
  RocketOutlined,
  SyncOutlined,
  ClockCircleOutlined,
} from "@ant-design/icons";
import * as echarts from "echarts";
import { api } from "@/services/api";
import { autoMiningApi } from "@/services/autoMining";
import dayjs from "dayjs";
import "./FactorMining.css";

const { Option } = Select;
const { RangePicker } = DatePicker;

interface Factor {
  id: number;
  name: string;
  code: string;
  category: string;
  source: "preset" | "user";
  description?: string;
}

interface MinedFactor {
  name: string;
  expression: string;
  ic: number;
  ir: number;
  fitness: number;
}

interface MiningStatus {
  task_id: string;
  status: "pending" | "running" | "completed" | "failed";
  current_generation: number;
  total_generations: number;
  best_fitness: number;
  avg_fitness: number;
  fitness_history?: {
    best: number[];
    average: number[];
  };
  error?: string;
}

interface MiningResult {
  factors: MinedFactor[];
  best_fitness: number;
  avg_fitness: number;
  generations: number;
  fitness_history?: {
    best: number[];
    average: number[];
  };
}

const FactorMining: React.FC = () => {
  const [form] = Form.useForm();
  const [activeTab, setActiveTab] = useState<"manual" | "auto">("manual");
  const evolutionChartRef = useRef<HTMLDivElement>(null);
  const resultChartRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<echarts.ECharts | null>(null);
  const resultChartInstanceRef = useRef<echarts.ECharts | null>(null);

  const [factors, setFactors] = useState<Factor[]>([]);
  const [loading, setLoading] = useState(false);
  const [mining, setMining] = useState(false);
  const [currentStockCode, setCurrentStockCode] = useState<string>("");

  const [miningStatus, setMiningStatus] = useState<MiningStatus | null>(null);
  const [miningResult, setMiningResult] = useState<MiningResult | null>(null);
  const [elapsedTime, setElapsedTime] = useState<number>(0); // 挖掘已用时间（秒）
  const miningStartTimeRef = useRef<number | null>(null); // 挖掘开始时间戳
  const elapsedTimeIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null); // 计时器ID
  const [, setSavedFactorNames] = useState<Set<string>>(
    new Set(),
  ); // 已保存的因子名称

  // 加载因子列表
  const loadFactors = async () => {
    try {
      const response = (await api.getFactors()) as any;
      if (response.success) {
        setFactors(response.data);
      }
    } catch (error) {
      console.error("加载因子列表失败:", error);
    }
  };

  useEffect(() => {
    loadFactors();

    // 设置默认日期范围
    const endDate = dayjs();
    const startDate = dayjs().subtract(1, "year");
    form.setFieldsValue({
      dateRange: [startDate, endDate],
      population_size: 50,
      n_generations: 10,
      mutation_rate: 0.2,
      crossover_rate: 0.7,
      elite_size: 5,
      fitness_objective: "ic_mean",
      ic_threshold: 0.03,
    });

    return () => {
      // 清理定时器
      if (window.miningInterval) {
        clearInterval(window.miningInterval);
      }
      // 清理计时器
      if (elapsedTimeIntervalRef.current) {
        clearInterval(elapsedTimeIntervalRef.current);
        elapsedTimeIntervalRef.current = null;
      }
      // 清理图表
      if (chartInstanceRef.current) {
        chartInstanceRef.current.dispose();
        chartInstanceRef.current = null;
      }
      if (resultChartInstanceRef.current) {
        resultChartInstanceRef.current.dispose();
        resultChartInstanceRef.current = null;
      }
    };
  }, []);

  // 开始挖掘
  const startMining = async (values: any) => {
    const selectedFactors = values.base_factors || [];

    // 保存当前股票代码，用于后续命名
    const stockCode = values.stock_code.replace(".", ""); // 移除股票代码中的点
    setCurrentStockCode(stockCode);

    const [startDate, endDate] = values.dateRange;
    const requestData = {
      stock_code: values.stock_code,
      base_factors: selectedFactors,
      start_date: startDate.format("YYYY-MM-DD"),
      end_date: endDate.format("YYYY-MM-DD"),
      population_size: values.population_size,
      n_generations: values.n_generations,
      cx_prob: values.crossover_rate,
      mut_prob: values.mutation_rate,
      elite_size: values.elite_size,
      fitness_objective: values.fitness_objective,
      ic_threshold: values.ic_threshold,
    };

    try {
      setLoading(true);
      setMining(true);
      setMiningResult(null);
      setElapsedTime(0); // 重置计时器
      setSavedFactorNames(new Set()); // 新挖掘时清除记录

      const response = (await autoMiningApi.startTask(requestData)) as any;

      if (response.success) {
        const newTaskId = response.data.task_id;

        // 记录挖掘开始时间
        miningStartTimeRef.current = Date.now();

        // 启动计时器，每秒更新一次已用时间
        elapsedTimeIntervalRef.current = setInterval(() => {
          if (miningStartTimeRef.current) {
            const elapsed = Math.floor(
              (Date.now() - miningStartTimeRef.current) / 1000,
            );
            setElapsedTime(elapsed);
          }
        }, 1000);

        // 轮询获取进度
        window.miningInterval = setInterval(() => {
          checkMiningProgress(newTaskId);
        }, 2000);

        message.success("挖掘任务已启动");
      }
    } catch (error) {
      console.error("启动挖掘失败:", error);
      message.error("启动挖掘失败");
      setMining(false);
    } finally {
      setLoading(false);
    }
  };

  // 检查挖掘进度
  const checkMiningProgress = async (currentTaskId: string) => {
    try {
      const response = (await autoMiningApi.getTaskStatus(currentTaskId)) as any;

      if (response.success) {
        const statusData = response.data as MiningStatus;
        setMiningStatus(statusData);

        console.log(
          "Mining status:",
          statusData.status,
          "Generation:",
          statusData.current_generation,
          "/",
          statusData.total_generations,
        );

        // 更新进化曲线 - 使用完整的历史数据
        if (
          statusData.fitness_history &&
          statusData.fitness_history.best.length > 0
        ) {
          console.log(
            "Updating evolution chart with history:",
            statusData.fitness_history,
          );
          updateEvolutionChart(statusData.fitness_history);
        } else if (statusData.current_generation > 0) {
          // 降级方案：如果没有历史数据，用当前值生成
          console.log("Using fallback for evolution chart");
          updateEvolutionChart({
            best: Array(statusData.current_generation).fill(
              statusData.best_fitness,
            ),
            average: Array(statusData.current_generation).fill(
              statusData.avg_fitness,
            ),
          });
        }

        // 检查是否完成
        if (statusData.status === "completed") {
          console.log(
            "Mining completed, clearing interval and getting results",
          );
          if (window.miningInterval) {
            clearInterval(window.miningInterval);
          }
          // 清除计时器
          if (elapsedTimeIntervalRef.current) {
            clearInterval(elapsedTimeIntervalRef.current);
            elapsedTimeIntervalRef.current = null;
          }
          miningStartTimeRef.current = null;
          setMining(false);
          await getMiningResults(currentTaskId);
        } else if (statusData.status === "failed") {
          console.log("Mining failed:", statusData.error);
          if (window.miningInterval) {
            clearInterval(window.miningInterval);
          }
          // 清除计时器
          if (elapsedTimeIntervalRef.current) {
            clearInterval(elapsedTimeIntervalRef.current);
            elapsedTimeIntervalRef.current = null;
          }
          miningStartTimeRef.current = null;
          setMining(false);
          message.error(`挖掘失败: ${statusData.error || "未知错误"}`);
        }
      }
    } catch (error) {
      console.error("获取进度失败:", error);
      // 不中断轮询，继续尝试获取进度
      // 只有在任务不存在时才停止
      if (error instanceof Error && error.message.includes("任务不存在")) {
        if (window.miningInterval) {
          clearInterval(window.miningInterval);
        }
        // 清除计时器
        if (elapsedTimeIntervalRef.current) {
          clearInterval(elapsedTimeIntervalRef.current);
          elapsedTimeIntervalRef.current = null;
        }
        miningStartTimeRef.current = null;
        setMining(false);
        message.error("任务不存在或已过期");
      }
    }
  };

  // 获取挖掘结果
  const getMiningResults = async (currentTaskId: string) => {
    try {
      console.log("Fetching mining results for task:", currentTaskId);
      const response = (await autoMiningApi.getTaskResult(currentTaskId)) as any;

      if (response.success) {
        console.log("Mining results received:", response.data);
        setMiningResult(response.data);

        // 绘制最终进化曲线（使用新的图表实例）
        if (response.data.fitness_history) {
          console.log(
            "Updating result chart with history:",
            response.data.fitness_history,
          );
          // 延迟一下确保DOM已渲染
          setTimeout(() => {
            updateResultChart(response.data.fitness_history);
          }, 200);
        }
      } else {
        message.error("获取结果失败: " + (response.message || "未知错误"));
      }
    } catch (error) {
      console.error("获取结果失败:", error);
      message.error("获取结果失败");
    }
  };

  // 更新进化曲线（进度中）
  const updateEvolutionChart = (fitnessHistory: {
    best: number[];
    average: number[];
  }) => {
    console.log(
      "updateEvolutionChart called, DOM element:",
      evolutionChartRef.current,
    );

    if (!evolutionChartRef.current) {
      console.error("Progress chart DOM element is null");
      return;
    }

    let chart = chartInstanceRef.current;
    if (!chart) {
      console.log("Initializing new progress chart instance");
      chart = echarts.init(evolutionChartRef.current);
      chartInstanceRef.current = chart;
    }

    const generations = fitnessHistory.best.map((_, i) => i + 1);

    const option = {
      title: {
        text: "进化曲线",
        left: "center",
        textStyle: { fontSize: 16, fontWeight: 600 },
      },
      tooltip: {
        trigger: "axis",
      },
      legend: {
        data: ["最优适应度", "平均适应度"],
        bottom: 0,
      },
      grid: {
        left: "3%",
        right: "4%",
        bottom: "10%",
        containLabel: true,
      },
      xAxis: {
        type: "category",
        name: "代数",
        data: generations,
      },
      yAxis: {
        type: "value",
        name: "适应度",
        scale: true,
      },
      series: [
        {
          name: "最优适应度",
          type: "line",
          data: fitnessHistory.best,
          smooth: true,
          itemStyle: { color: "#3b82f6" },
          areaStyle: {
            color: {
              type: "linear",
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: "rgba(59, 130, 246, 0.3)" },
                { offset: 1, color: "rgba(59, 130, 246, 0.05)" },
              ],
            },
          },
        },
        {
          name: "平均适应度",
          type: "line",
          data: fitnessHistory.average,
          smooth: true,
          itemStyle: { color: "#22c55e" },
          areaStyle: {
            color: {
              type: "linear",
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: "rgba(34, 197, 94, 0.3)" },
                { offset: 1, color: "rgba(34, 197, 94, 0.05)" },
              ],
            },
          },
        },
      ],
    };

    try {
      chart.setOption(option, true);
      console.log("Progress chart updated successfully");
    } catch (error) {
      console.error("Error updating progress chart:", error);
    }
  };

  // 更新最终结果图表
  const updateResultChart = (fitnessHistory: {
    best: number[];
    average: number[];
  }) => {
    console.log(
      "updateResultChart called, DOM element:",
      resultChartRef.current,
    );

    if (!resultChartRef.current) {
      console.error("Result chart DOM element is null");
      return;
    }

    let chart = resultChartInstanceRef.current;
    if (!chart) {
      console.log("Initializing new result chart instance");
      chart = echarts.init(resultChartRef.current);
      resultChartInstanceRef.current = chart;
    }

    const generations = fitnessHistory.best.map((_, i) => i + 1);

    const option = {
      title: {
        text: "完整进化曲线",
        left: "center",
        textStyle: { fontSize: 16, fontWeight: 600 },
      },
      tooltip: {
        trigger: "axis",
      },
      legend: {
        data: ["最优适应度", "平均适应度"],
        bottom: 0,
      },
      grid: {
        left: "3%",
        right: "4%",
        bottom: "10%",
        containLabel: true,
      },
      xAxis: {
        type: "category",
        name: "代数",
        data: generations,
      },
      yAxis: {
        type: "value",
        name: "适应度",
        scale: true,
      },
      series: [
        {
          name: "最优适应度",
          type: "line",
          data: fitnessHistory.best,
          smooth: true,
          itemStyle: { color: "#3b82f6" },
          areaStyle: {
            color: {
              type: "linear",
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: "rgba(59, 130, 246, 0.3)" },
                { offset: 1, color: "rgba(59, 130, 246, 0.05)" },
              ],
            },
          },
        },
        {
          name: "平均适应度",
          type: "line",
          data: fitnessHistory.average,
          smooth: true,
          itemStyle: { color: "#22c55e" },
          areaStyle: {
            color: {
              type: "linear",
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: "rgba(34, 197, 94, 0.3)" },
                { offset: 1, color: "rgba(34, 197, 94, 0.05)" },
              ],
            },
          },
        },
      ],
    };

    try {
      chart.setOption(option, true);
      console.log("Result chart updated successfully");
    } catch (error) {
      console.error("Error updating result chart:", error);
    }
  };

  // 保存单个因子（带重试机制）
  const saveFactor = async (
    factor: MinedFactor,
    index: number,
    retryCount: number = 0,
  ) => {
    // 生成因子名称：Mined_Factor_序号_年月日时分秒_股票代码
    const today = new Date();
    const dateStr = [
      today.getFullYear(),
      String(today.getMonth() + 1).padStart(2, "0"),
      String(today.getDate()).padStart(2, "0"),
      String(today.getHours()).padStart(2, "0"),
      String(today.getMinutes()).padStart(2, "0"),
      String(today.getSeconds()).padStart(2, "0"),
    ].join("");

    // 确保使用有效的股票代码
    const stockCode = currentStockCode || "Unknown";
    const baseFactorName = `Mined_Factor_${index + 1}_${dateStr}_${stockCode}`;

    // 根据重试次数生成名称
    let factorName: string;
    if (retryCount === 0) {
      factorName = baseFactorName;
    } else {
      factorName = `${baseFactorName}_${retryCount}`;
    }

    try {
      // 将表达式包装成完整的函数
      const generateFactorFunction = (expr: string) => {
        // 为表达式添加 df 前缀
        const processedExpr = expr
          .replace(/\bopen\b/g, "df['open']")
          .replace(/\bclose\b/g, "df['close']")
          .replace(/\bhigh\b/g, "df['high']")
          .replace(/\blow\b/g, "df['low']")
          .replace(/\bvolume\b/g, "df['volume']");

        return `def calculate_factor(df):
    """
    遗传算法挖掘因子
    表达式: ${expr}
    IC: ${factor.ic?.toFixed(4)}
    IR: ${factor.ir?.toFixed(4)}
    """
    import pandas as pd
    import numpy as np

    try:
        result = ${processedExpr}
        return result
    except Exception as e:
        # 如果计算失败，返回全0序列
        return pd.Series(0, index=df.index)
`;
      };

      const factorData = {
        name: factorName,
        code: generateFactorFunction(factor.expression),
        category: "遗传挖掘",
        description: `通过遗传算法挖掘的因子 | 表达式: ${factor.expression} | IC: ${factor.ic?.toFixed(4)} | IR: ${factor.ir?.toFixed(4)} | 适应度: ${factor.fitness?.toFixed(4)}`,
        formula_type: "function",
      };

      console.log("Saving factor:", factorData);
      console.log("Factor code length:", factorData.code.length);

      const response = (await api.createFactor(factorData)) as any;

      if (response.success) {
        message.success(`因子 "${factorName}" 已保存到自定义因子库`);
        // 记录已保存的因子
        setSavedFactorNames((prev) => new Set(prev).add(factorName));
        // 刷新因子列表
        await loadFactors();
      } else {
        message.error(
          "保存失败: " +
            (response.data?.detail || response.message || "未知错误"),
        );
      }
    } catch (error: any) {
      console.error("保存因子失败:", error);
      const errorMsg =
        error.response?.data?.detail ||
        error.response?.data?.message ||
        error.message ||
        "未知错误";

      // 如果是"已存在"错误且重试次数少于5次，使用新名称重试
      if (errorMsg.includes("已存在") && retryCount < 5) {
        console.log(
          `因子名称 ${factorName} 已存在，尝试使用新名称 (重试 ${retryCount + 1}/5)`,
        );
        await saveFactor(factor, index, retryCount + 1);
      } else {
        message.error("保存因子失败: " + errorMsg);
      }
    }
  };

  // 保存全部因子
  const saveAllFactors = async () => {
    if (
      !miningResult ||
      !miningResult.factors ||
      miningResult.factors.length === 0
    ) {
      message.warning("没有可保存的因子");
      return;
    }

    // 生成日期字符串（包含时分秒）
    const today = new Date();
    const dateStr = [
      today.getFullYear(),
      String(today.getMonth() + 1).padStart(2, "0"),
      String(today.getDate()).padStart(2, "0"),
      String(today.getHours()).padStart(2, "0"),
      String(today.getMinutes()).padStart(2, "0"),
      String(today.getSeconds()).padStart(2, "0"),
    ].join("");

    // 确保使用有效的股票代码
    const stockCode = currentStockCode || "Unknown";

    let successCount = 0;
    let failCount = 0;

    for (let i = 0; i < miningResult.factors.length; i++) {
      const factor = miningResult.factors[i];

      // 直接生成唯一的因子名称（包含序号、日期时间、股票代码）
      const factorName = `Mined_Factor_${i + 1}_${dateStr}_${stockCode}`;

      try {
        // 生成完整的因子函数代码
        const processedExpr = factor.expression
          .replace(/\bopen\b/g, "df['open']")
          .replace(/\bclose\b/g, "df['close']")
          .replace(/\bhigh\b/g, "df['high']")
          .replace(/\blow\b/g, "df['low']")
          .replace(/\bvolume\b/g, "df['volume']");

        const factorCode = `def calculate_factor(df):
    """
    遗传算法挖掘因子
    表达式: ${factor.expression}
    IC: ${factor.ic?.toFixed(4)}
    IR: ${factor.ir?.toFixed(4)}
    """
    import pandas as pd
    import numpy as np

    try:
        result = ${processedExpr}
        return result
    except Exception as e:
        return pd.Series(0, index=df.index)
`;

        const factorData = {
          name: factorName,
          code: factorCode,
          category: "遗传挖掘",
          description: `通过遗传算法挖掘的因子 | 表达式: ${factor.expression} | IC: ${factor.ic?.toFixed(4)} | IR: ${factor.ir?.toFixed(4)} | 适应度: ${factor.fitness?.toFixed(4)}`,
          formula_type: "function",
        };

        const response = (await api.createFactor(factorData)) as any;

        if (response.success) {
          successCount++;
          message.success(`因子 "${factorName}" 已保存到自定义因子库`);
          setSavedFactorNames((prev) => new Set(prev).add(factorName));
          await loadFactors();
        } else {
          failCount++;
          message.error(
            "保存失败: " +
              (response.data?.detail || response.message || "未知错误"),
          );
        }
      } catch (error: any) {
        failCount++;
        console.error("保存因子失败:", error);
        const errorMsg =
          error.response?.data?.detail ||
          error.response?.data?.message ||
          error.message ||
          "未知错误";
        message.error(`保存因子失败: ${errorMsg}`);
      }
    }

    // 显示结果消息
    if (failCount === 0) {
      message.success(`成功保存 ${successCount} 个因子到自定义因子库`);
    } else {
      message.warning(
        `保存完成: 成功 ${successCount} 个, 失败 ${failCount} 个`,
      );
    }
  };

  // 计算进度百分比
  const getProgressPercent = () => {
    if (!miningStatus || miningStatus.total_generations === 0) return 0;
    return Math.round(
      (miningStatus.current_generation / miningStatus.total_generations) * 100,
    );
  };

  // 格式化挖掘时长
  const formatElapsedTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;

    if (hours > 0) {
      return `${hours}小时${minutes}分${secs}秒`;
    } else if (minutes > 0) {
      return `${minutes}分${secs}秒`;
    } else {
      return `${secs}秒`;
    }
  };

  const manualMiningContent = (
    <Row gutter={[24, 24]}>
      <Col xs={24} lg={8}>
        <Card title="遗传算法配置" className="config-card">
          <Form form={form} layout="vertical" onFinish={startMining}>
            <Divider
              styles={{ content: { margin: 0 } }}
              titlePlacement="left"
            >
              基础配置
            </Divider>

            <Form.Item
              label="股票代码"
              name="stock_code"
              initialValue="000001"
              rules={[{ required: true, message: "请输入股票代码" }]}
            >
              <Input placeholder="例如：000001、600000" />
            </Form.Item>

            <Form.Item
              label="日期范围"
              name="dateRange"
              rules={[{ required: true, message: "请选择日期范围" }]}
            >
              <RangePicker style={{ width: "100%" }} />
            </Form.Item>

            <Divider
              styles={{ content: { margin: 0 } }}
              titlePlacement="left"
            >
              基础因子选择
            </Divider>
            <p className="text-hint">
              选择作为遗传算法输入的基础因子（可搜索因子名称）
            </p>

            <Form.Item
              name="base_factors"
              rules={[{ required: true, message: "请至少选择一个基础因子" }]}
            >
              <Select
                mode="multiple"
                placeholder="输入因子名称搜索，如：RSI、MACD、SMA"
                style={{ width: "100%" }}
                showSearch
                filterOption={(input, option) => {
                  const label = String(option?.label ?? "");
                  const value = String(option?.value ?? "");
                  return (
                    label.toLowerCase().includes(input.toLowerCase()) ||
                    value.toLowerCase().includes(input.toLowerCase())
                  );
                }}
                optionLabelProp="label"
                maxTagCount="responsive"
                size="large"
                popupClassName="factor-select-dropdown"
                listHeight={400}
              >
                {factors.map((factor) => (
                  <Option
                    key={factor.id}
                    value={factor.name}
                    label={factor.name}
                  >
                    <div
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        gap: 4,
                      }}
                    >
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: 8,
                        }}
                      >
                        <span style={{ fontWeight: 500 }}>{factor.name}</span>
                        <Tag
                          color={
                            factor.source === "preset" ? "success" : "warning"
                          }
                        >
                          {factor.source === "preset" ? "预置" : "自定义"}
                        </Tag>
                        <Tag color="blue">{factor.category}</Tag>
                      </div>
                      <div
                        style={{
                          fontSize: 12,
                          color: "#64748b",
                          fontFamily: "monospace",
                        }}
                      >
                        {factor.code}
                      </div>
                      {factor.description && (
                        <div style={{ fontSize: 12, color: "#94a3b8" }}>
                          {factor.description}
                        </div>
                      )}
                    </div>
                  </Option>
                ))}
              </Select>
            </Form.Item>

            <Form.Item noStyle shouldUpdate>
              {() => {
                const selectedCount =
                  form.getFieldValue("base_factors")?.length || 0;
                return (
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      marginBottom: 16,
                    }}
                  >
                    <span className="text-hint">
                      已选择 <strong style={{ color: "#3b82f6" }}>{selectedCount}</strong> 个因子
                    </span>
                    <Space size="small">
                      <Button
                        type="link"
                        size="small"
                        onClick={() => {
                          form.setFieldsValue({
                            base_factors: factors.map((f) => f.name),
                          });
                        }}
                      >
                        全选
                      </Button>
                      <Button
                        type="link"
                        size="small"
                        onClick={() => {
                          form.setFieldsValue({ base_factors: [] });
                        }}
                      >
                        清空
                      </Button>
                    </Space>
                  </div>
                );
              }}
            </Form.Item>

            <Divider
              styles={{ content: { margin: 0 } }}
              titlePlacement="left"
            >
              算法参数
            </Divider>

            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  label="种群大小"
                  name="population_size"
                  tooltip="每一代的个体数量"
                >
                  <InputNumber min={10} max={200} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item
                  label="迭代次数"
                  name="n_generations"
                  tooltip="进化代数"
                >
                  <InputNumber min={1} max={100} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
            </Row>

            <Row gutter={16}>
              <Col span={12}>
                <Form.Item label="变异率" name="mutation_rate">
                  <InputNumber
                    min={0}
                    max={1}
                    step={0.05}
                    style={{ width: "100%" }}
                  />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item label="交叉率" name="crossover_rate">
                  <InputNumber
                    min={0}
                    max={1}
                    step={0.05}
                    style={{ width: "100%" }}
                  />
                </Form.Item>
              </Col>
            </Row>

            <Form.Item
              label="精英保留数量"
              name="elite_size"
              tooltip="每代保留的最优个体数"
            >
              <InputNumber min={0} max={20} style={{ width: "100%" }} />
            </Form.Item>

            <Divider
              styles={{ content: { margin: 0 } }}
              titlePlacement="left"
            >
              适应度函数
            </Divider>

            <Form.Item label="优化目标" name="fitness_objective">
              <Select>
                <Option value="ic_mean">IC均值</Option>
                <Option value="ir_ratio">IR比率</Option>
                <Option value="sharpe">夏普比率</Option>
                <Option value="combined">综合得分</Option>
              </Select>
            </Form.Item>

            <Form.Item noStyle shouldUpdate>
              {() => {
                const objective =
                  form.getFieldValue("fitness_objective") || "ic_mean";

                let thresholdLabel = "阈值";
                let thresholdPlaceholder = "0.03";

                if (objective === "ic_mean") {
                  thresholdLabel = "IC阈值";
                  thresholdPlaceholder = "例如：0.03";
                } else if (objective === "ir_ratio") {
                  thresholdLabel = "IR阈值";
                  thresholdPlaceholder = "例如：0.5";
                } else if (objective === "sharpe") {
                  thresholdLabel = "夏普阈值";
                  thresholdPlaceholder = "例如：1.0";
                } else if (objective === "combined") {
                  thresholdLabel = "综合阈值";
                  thresholdPlaceholder = "例如：0.5";
                }

                return (
                  <Form.Item
                    label={thresholdLabel}
                    name="ic_threshold"
                    tooltip={`筛选因子的${thresholdLabel}`}
                  >
                    <InputNumber
                      min={0}
                      step={0.01}
                      style={{ width: "100%" }}
                      placeholder={thresholdPlaceholder}
                    />
                  </Form.Item>
                );
              }}
            </Form.Item>

            <Form.Item>
              <Button
                type="primary"
                htmlType="submit"
                icon={<PlayCircleOutlined />}
                loading={loading}
                block
                size="large"
                disabled={mining}
              >
                {mining ? "挖掘中..." : "开始挖掘"}
              </Button>
            </Form.Item>
          </Form>
        </Card>
      </Col>

      <Col xs={24} lg={16}>
        <Card title="挖掘结果" className="result-card">
          {!mining && !miningStatus && !miningResult && (
            <div className="placeholder-content">
              <BarChartOutlined className="placeholder-icon" />
              <p className="placeholder-text">配置参数后点击"开始挖掘"按钮</p>
              <p className="placeholder-hint">
                遗传算法将自动搜索最优因子表达式
              </p>
            </div>
          )}

          {(mining || miningStatus) && !miningResult && (
            <div className="mining-progress">
              {mining && (
                <Alert
                  message={
                    <Space>
                      <SyncOutlined spin />
                      <span>挖掘进行中...</span>
                      <ClockCircleOutlined />
                      <span style={{ color: "#64748b" }}>
                        已用时: {formatElapsedTime(elapsedTime)}
                      </span>
                    </Space>
                  }
                  type="info"
                  showIcon={false}
                  style={{
                    marginBottom: 16,
                    background: "rgba(59, 130, 246, 0.1)",
                    border: "1px solid rgba(59, 130, 246, 0.2)",
                  }}
                />
              )}

              {miningStatus && (
                <>
                  <div className="progress-section">
                    <div className="progress-header">
                      <span className="progress-label">挖掘进度</span>
                      <span className="progress-value">
                        {getProgressPercent()}%
                      </span>
                    </div>
                    <Progress
                      percent={getProgressPercent()}
                      status={mining ? "active" : "success"}
                      strokeColor={{
                        "0%": "#3b82f6",
                        "100%": "#22c55e",
                      }}
                    />
                  </div>

                  <Row gutter={16} style={{ marginTop: 24 }}>
                    <Col span={8}>
                      <div className="stat-item">
                        <p className="stat-label">当前代数</p>
                        <p className="stat-value">
                          {miningStatus.current_generation}/
                          {miningStatus.total_generations}
                        </p>
                      </div>
                    </Col>
                    <Col span={8}>
                      <div className="stat-item">
                        <p className="stat-label">最优适应度</p>
                        <p className="stat-value stat-primary">
                          {miningStatus.best_fitness?.toFixed(4) || "-"}
                        </p>
                      </div>
                    </Col>
                    <Col span={8}>
                      <div className="stat-item">
                        <p className="stat-label">平均适应度</p>
                        <p className="stat-value">
                          {miningStatus.avg_fitness?.toFixed(4) || "-"}
                        </p>
                      </div>
                    </Col>
                  </Row>
                </>
              )}

              {mining && !miningStatus && (
                <div
                  style={{
                    textAlign: "center",
                    padding: "40px 0",
                    color: "#64748b",
                  }}
                >
                  <Spin size="large" />
                  <p style={{ marginTop: 16 }}>正在执行挖掘任务...</p>
                </div>
              )}

              {miningStatus && (
                <div className="chart-section" style={{ marginTop: 24 }}>
                  <h4 className="chart-title">进化曲线（实时）</h4>
                  <div
                    ref={evolutionChartRef}
                    className="chart-container"
                    style={{ height: "300px" }}
                  ></div>
                </div>
              )}
            </div>
          )}

          {miningStatus &&
            miningStatus.status === "completed" &&
            !miningResult && (
              <div style={{ textAlign: "center", padding: "24px" }}>
                <Spin size="large" tip="正在加载挖掘结果..." />
              </div>
            )}

          {miningResult && (
            <div className="mining-result">
              <div className="result-summary" style={{ marginBottom: 24 }}>
                <Row gutter={16}>
                  <Col span={8}>
                    <div className="stat-item">
                      <p className="stat-label">总代数</p>
                      <p className="stat-value">{miningResult.generations}</p>
                    </div>
                  </Col>
                  <Col span={8}>
                    <div className="stat-item">
                      <p className="stat-label">最优适应度</p>
                      <p className="stat-value stat-primary">
                        {miningResult.best_fitness?.toFixed(4)}
                      </p>
                    </div>
                  </Col>
                  <Col span={8}>
                    <div className="stat-item">
                      <p className="stat-label">发现因子数</p>
                      <p className="stat-value">
                        {miningResult.factors?.length || 0}
                      </p>
                    </div>
                  </Col>
                </Row>
              </div>

              <div className="chart-section" style={{ marginBottom: 24 }}>
                <h4 className="chart-title">完整进化曲线</h4>
                <div
                  ref={resultChartRef}
                  className="chart-container"
                  style={{ height: "300px" }}
                ></div>
              </div>

              <Divider />

              <h3 className="result-title">发现的因子</h3>

              {!miningResult.factors || miningResult.factors.length === 0 ? (
                <Alert
                  message="未发现符合条件的因子"
                  type="info"
                  showIcon
                  style={{ marginTop: 16 }}
                />
              ) : (
                <div className="factors-list">
                  {miningResult.factors.map((factor, index) => (
                    <Card key={index} className="factor-card" size="small">
                      <div className="factor-header">
                        <div className="factor-info">
                          <Space>
                            <Tag color="blue">Top {index + 1}</Tag>
                            <span className="factor-name">
                              {factor.name || `Factor_${index + 1}`}
                            </span>
                          </Space>
                          <div className="factor-expression">
                            {factor.expression}
                          </div>
                        </div>
                        <div className="factor-stats">
                          <div className="stat-row">
                            <span className="stat-label">IC:</span>
                            <span
                              className={`stat-value ${factor.ic > 0 ? "positive" : "negative"}`}
                            >
                              {factor.ic?.toFixed(4)}
                            </span>
                          </div>
                          <div className="stat-row">
                            <span className="stat-label">IR:</span>
                            <span
                              className={`stat-value ${factor.ir > 0 ? "positive" : "negative"}`}
                            >
                              {factor.ir?.toFixed(4)}
                            </span>
                          </div>
                        </div>
                      </div>
                      <div className="factor-actions">
                        <Button
                          type="primary"
                          size="small"
                          icon={<SaveOutlined />}
                          onClick={() => saveFactor(factor, index)}
                        >
                          保存到因子库
                        </Button>
                      </div>
                    </Card>
                  ))}
                </div>
              )}

              <div className="result-actions" style={{ marginTop: 24 }}>
                <Space>
                  <Button
                    type="primary"
                    icon={<SaveOutlined />}
                    onClick={saveAllFactors}
                  >
                    全部保存到因子库
                  </Button>
                </Space>
              </div>
            </div>
          )}
        </Card>
      </Col>
    </Row>
  );

  const autoMiningContent = (
    <Card className="result-card">
      <Space direction="vertical" size={16} style={{ width: "100%" }}>
        <Alert
          type="info"
          showIcon
          message="自动挖掘入口迁移中"
          description="本批先完成 QuantGPT 风格的 API 封装落点，以及页面 Tab 结构对齐。下一批将接入真实 task 提交、状态轮询和候选展示。"
        />
        <div className="placeholder-content" style={{ padding: "24px 0" }}>
          <RocketOutlined className="placeholder-icon" />
          <p className="placeholder-text">自动挖掘 Tab 已就位</p>
          <p className="placeholder-hint">
            后续会以 QuantGPT 的异步 task 流程为准，逐步迁入提交、状态、候选结果与迭代动作。
          </p>
        </div>
      </Space>
    </Card>
  );

  return (
    <div className="factor-mining-container">
      {/* 背景 */}
      <div className="bg-gradient"></div>
      <div className="bg-grid"></div>

      <div className="factor-mining-content">
        <div className="page-header">
          <div className="header-content">
            <RocketOutlined className="header-icon" />
            <div>
              <h1 className="page-title">因子挖掘</h1>
              <p className="page-subtitle">
                以 QuantGPT 的真实任务流为准，逐步迁移 auto mining 能力
              </p>
            </div>
          </div>
        </div>
        <Tabs
          activeKey={activeTab}
          onChange={(key) => setActiveTab(key as "manual" | "auto")}
          items={[
            {
              key: "manual",
              label: "手动挖掘",
              children: manualMiningContent,
            },
            {
              key: "auto",
              label: "自动挖掘",
              children: autoMiningContent,
            },
          ]}
        />
      </div>
    </div>
  );
};

export default FactorMining;
