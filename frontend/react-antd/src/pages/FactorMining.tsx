import { useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Checkbox,
  Col,
  Collapse,
  DatePicker,
  Divider,
  Form,
  Input,
  InputNumber,
  Progress,
  Row,
  Segmented,
  Select,
  Space,
  Tabs,
  Tag,
  message,
} from "antd";
import {
  ExclamationCircleOutlined,
  BarChartOutlined,
  PlayCircleOutlined,
  RocketOutlined,
  SaveOutlined,
  SettingOutlined,
  FileSearchOutlined,
  OrderedListOutlined,
} from "@ant-design/icons";
import * as echarts from "echarts";
import dayjs from "dayjs";
import { api } from "@/services/api";
import { autoMiningApi } from "@/services/autoMining";
import { resolveApiUrl } from "@/services/url";
import QuantTaskDetailsPanel from "./FactorTaskDetailsPanel";
import "./FactorMining.css";

const { RangePicker } = DatePicker;
const { Option } = Select;
const { Panel } = Collapse;

const AUTO_MINING_FORM_STORAGE_KEY = "factorhub:auto-mining-form";
const RDAGENT_ACTIVE_TASK_STORAGE_KEY = "factorhub:rdagent-active-task";
const AUTO_PROGRESS_POLL_MS = 1000;
const RDAGENT_TERMINAL_STATUSES = ["completed", "failed", "cancelled"] as const;
const RDAGENT_RESUMABLE_STATUSES = ["pending", "running", "cancelling"] as const;
const FACTORHUB_MARKET_FIELDS = ["open", "high", "low", "close", "volume", "amount", "vwap", "pct_change"];
const OPTIMIZATION_DIRECTION_OPTIONS = [
  { label: "优化 Score", value: "score" },
  { label: "优化 L/S Sharpe", value: "ls_sharpe" },
  { label: "优化 L/S Return", value: "ls_return" },
  { label: "优化 WQ Rating", value: "wq_rating" },
  { label: "优化 WQ Fitness", value: "wq_fitness" },
  { label: "优化 WQ Return", value: "wq_return" },
  { label: "优化 Report Sharpe", value: "report_sharpe" },
];

type MiningMode = "manual" | "auto" | "rdagent";
type RDAgentBootstrapMode = "manual" | "llm_auto";

interface Factor {
  id: number;
  name: string;
  code: string;
  category: string;
  source: "preset" | "user";
  description?: string;
}

interface ManualFactor {
  name: string;
  expression: string;
  ic: number;
  ir: number;
  fitness: number;
}

interface AutoFactor {
  name: string;
  expression: string;
  score: number;
  grade?: string;
  report_url?: string;
  report_metrics?: Record<string, any>;
  backtest_summary?: Record<string, any>;
  component_scores?: Record<string, any>;
  anti_overfit?: Record<string, any>;
  wq_brain?: Record<string, any>;
  interpretation?: Record<string, any>;
  task_details?: Record<string, any>;
  quantgpt_task_details?: Record<string, any>; // legacy compatibility
  task_id?: string;
  source_expression?: string;
  source?: string;
}

const getFactorDetailKey = (factor: AutoFactor, index: number) =>
  `${factor.task_id || factor.name || "factor"}-${index}`;

interface MiningStatus {
  task_id: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  current_generation: number;
  total_generations: number;
  best_fitness: number;
  avg_fitness: number;
  fitness_history?: { best: number[]; average: number[] };
  error?: string;
  candidates?: AutoFactor[];
}

interface ManualMiningResult {
  factors: ManualFactor[];
  best_fitness: number;
  avg_fitness: number;
  generations: number;
  fitness_history?: { best: number[]; average: number[] };
}

interface AutoMiningResult {
  factors: AutoFactor[];
  parent_factor?: AutoFactor;
  candidates?: AutoFactor[];
  best_score: number;
  avg_score: number;
  generations: number;
  fitness_history?: { best: number[]; average: number[] };
  upstream?: Record<string, string>;
  round_evaluation?: Record<string, any>;
}

interface AutoCampaignStatus extends MiningStatus {
  progress: number;
  current_round: number;
  total_rounds: number;
  retained_count: number;
  upstream_status?: string;
  cancel_requested?: boolean;
  rounds?: RDAgentTraceRound[];
  latest_round?: RDAgentTraceRound;
  manual_report?: Record<string, any>;
  continue_mining_request?: Record<string, any>;
}

interface AutoCampaignRoundSummary {
  round_index: number;
  task_id: string;
  best_score: number;
  avg_score: number;
  input_base_factors: string[];
  previous_base_factors?: string[];
  factor_changes?: {
    added?: string[];
    removed?: string[];
    retained?: string[];
  };
  factor_update_mode?: "initial" | "append" | "reselect";
  selected_factors?: string[];
  selection_rationale?: string;
  per_factor_reason?: Record<string, string>;
  continuation_hypothesis?: {
    hypothesis?: string;
    reason?: string;
    target_goal?: string;
    primary_problem?: string;
    current_base_factors?: string[];
    candidate_factors?: string[];
    factor_update_mode?: string;
  };
  continuation_feedback?: {
    decision?: boolean;
    accepted_as_best?: boolean;
    observations?: string;
    hypothesis_evaluation?: string;
    reason?: string;
    score_delta?: number;
    parent_best_score?: number;
    current_best_score?: number;
    next_hypothesis?: string;
  };
  retained_count: number;
  retained_factors: AutoFactor[];
  all_factors: AutoFactor[];
}

interface AutoCampaignResult {
  rounds: AutoCampaignRoundSummary[];
  retained_factors: AutoFactor[];
  final_round_task_id?: string;
  final_round_result?: AutoMiningResult;
  best_score: number;
  avg_score: number;
  fitness_history?: { best: number[]; average: number[] };
  selection_mode?: "all" | "any";
  retention_filter?: Record<string, any>;
  manual_report?: Record<string, any>;
  continue_mining_request?: Record<string, any>;
}

interface RDAgentTraceRound {
  round_index: number;
  task_id: string;
  hypothesis?: Record<string, any>;
  candidates?: AutoFactor[];
  evaluation?: Record<string, any>;
  feedback?: Record<string, any>;
  best_score: number;
  avg_score: number;
  input_base_factors?: string[];
  retained_count: number;
  retained_factors?: AutoFactor[];
  all_factors?: AutoFactor[];
  manual_report?: Record<string, any>;
  continue_mining_request?: Record<string, any>;
}

interface AutoMiningSeedState {
  result: AutoMiningResult | null;
  status: MiningStatus | null;
}

const buildManualChartHistory = (
  status?: Pick<MiningStatus, "fitness_history" | "current_generation" | "best_fitness" | "avg_fitness"> | null,
) => {
  const history = status?.fitness_history;
  if (history && ((history.best?.length || 0) > 0 || (history.average?.length || 0) > 0)) {
    return history;
  }
  const generationCount = Number(status?.current_generation || 0);
  if (generationCount <= 0) return undefined;
  return {
    best: Array(generationCount).fill(Number(status?.best_fitness || 0)),
    average: Array(generationCount).fill(Number(status?.avg_fitness || 0)),
  };
};

interface ContinueExplorationValues {
  prompt?: string;
  direction?: string;
  additional_base_factors?: string[];
  auto_additional_factor_count?: number;
  n_candidates?: number;
  n_groups?: number;
  holding_period?: number;
  neutralize_industry?: boolean;
  neutralize_cap?: boolean;
}

interface ContinuationInsightSummary {
  baseFactors: string[];
  weaknesses: string[];
  optimizationDirections: string[];
}

interface PersistedRDAgentTaskState {
  taskId: string;
  status?: string;
  updatedAt?: string;
}

const normalizeTaskDetailsForDashboard = (raw?: Record<string, any> | null) => {
  if (!raw || typeof raw !== 'object') return null;
  const reportMetrics = raw.report_metrics || raw.metrics || {};
  const quantgptReportUrl =
    raw.quantgpt_report_url
    || raw.manual_report?.html_report_url
    || raw.manual_report?.location
    || raw.raw_report_ref;
  const factorhubReportUrl = raw.factorhub_report_url || raw.report_url;
  return {
    ...raw,
    report_metrics: reportMetrics,
    metrics: reportMetrics,
    expression: raw.expression || raw.params?.expression || raw.llm?.generated_expression,
    report_url: factorhubReportUrl,
    factorhub_report_url: factorhubReportUrl,
    quantgpt_report_url: quantgptReportUrl,
    backtest_summary: raw.backtest_summary || {},
    wq_brain: raw.wq_brain || {},
    interpretation: raw.interpretation || {},
    round_evaluation: raw.round_evaluation || {},
    params: raw.params || {},
    llm: raw.llm || {},
  };
};

const buildFactorTaskDetailsForDashboard = (factor: Partial<AutoFactor> | null | undefined) => {
  if (!factor || typeof factor !== "object") return null;
  const rawDetails = (factor.task_details || factor.quantgpt_task_details || {}) as Record<string, any>;
  return normalizeTaskDetailsForDashboard({
    ...rawDetails,
    expression: rawDetails.expression || factor.expression || factor.source_expression,
    report_url: rawDetails.report_url || factor.report_url,
    factorhub_report_url: rawDetails.factorhub_report_url || rawDetails.report_url || factor.report_url,
    quantgpt_report_url:
      rawDetails.quantgpt_report_url
      || rawDetails.manual_report?.html_report_url
      || rawDetails.manual_report?.location
      || rawDetails.raw_report_ref,
    report_metrics: rawDetails.report_metrics || rawDetails.metrics || factor.report_metrics || {},
    metrics: rawDetails.metrics || rawDetails.report_metrics || factor.report_metrics || {},
    backtest_summary: rawDetails.backtest_summary || factor.backtest_summary || {},
    wq_brain: rawDetails.wq_brain || factor.wq_brain || {},
    anti_overfit: rawDetails.anti_overfit || factor.anti_overfit || {},
    interpretation: rawDetails.interpretation || factor.interpretation || {},
    round_evaluation: rawDetails.round_evaluation || {},
    scoring: rawDetails.scoring || {
      score: factor.score,
      grade: factor.grade,
      component_scores: factor.component_scores || {},
      wq_fitness: factor.wq_brain?.wq_fitness,
    },
  });
};

const normalizeRDAgentFactorForDashboard = (factor: any, roundIndex?: number, taskId?: string): AutoFactor => {
  const rawDetails = factor?.task_details || factor?.quantgpt_task_details || {};
  const rdagentScore = rawDetails?.rdagent?.candidate_score || {};
  const reportMetrics = factor?.report_metrics || rdagentScore.report_metrics || rawDetails.report_metrics || {};
  const backtestSummary = factor?.backtest_summary || rdagentScore.backtest_summary || rawDetails.backtest_summary || {};
  const details = normalizeTaskDetailsForDashboard({
    ...rawDetails,
    expression: rawDetails.expression || factor?.expression,
    report_url: rawDetails.report_url || factor?.report_url || rdagentScore.report_url,
    report_metrics: reportMetrics,
    metrics: rawDetails.metrics || reportMetrics,
    backtest_summary: backtestSummary,
    rdagent: rawDetails.rdagent || {},
    scoring: rawDetails.scoring || {
      score: factor?.score,
      grade: factor?.grade,
      component_scores: factor?.component_scores || {},
    },
  });
  return {
    ...factor,
    report_url: factor?.report_url || details?.report_url || rdagentScore.report_url,
    report_metrics: reportMetrics,
    backtest_summary: backtestSummary,
    task_details: details || rawDetails,
    quantgpt_task_details: details || rawDetails,
    automation_meta: {
      ...(factor?.automation_meta || {}),
      round_index: factor?.automation_meta?.round_index ?? roundIndex,
      round_task_id: factor?.automation_meta?.round_task_id || taskId,
      source: factor?.automation_meta?.source || "rdagent",
    },
  };
};

const normalizeRDAgentResultForDashboard = (result: AutoCampaignResult | null | undefined): AutoCampaignResult | null => {
  if (!result) return null;
  const rounds = (result.rounds || []).map((round: any) => {
    const factors = (round.candidates || round.all_factors || []).map((factor: any) =>
      normalizeRDAgentFactorForDashboard(factor, round.round_index, round.task_id)
    );
    return {
      ...round,
      candidates: factors,
      all_factors: factors,
      retained_factors: (round.retained_factors || factors.filter((factor: any) => factor.status === "accepted")).map((factor: any) =>
        normalizeRDAgentFactorForDashboard(factor, round.round_index, round.task_id)
      ),
    };
  });
  const finalRound = rounds[rounds.length - 1] as any;
  return {
    ...result,
    rounds,
    retained_factors: (result.retained_factors || []).map((factor: any) =>
      normalizeRDAgentFactorForDashboard(
        factor,
        factor?.automation_meta?.round_index,
        factor?.automation_meta?.round_task_id || finalRound?.task_id,
      )
    ),
    final_round_result: result.final_round_result
      ? {
          ...result.final_round_result,
          factors: (result.final_round_result.factors || finalRound?.candidates || []).map((factor: any) =>
            normalizeRDAgentFactorForDashboard(factor, finalRound?.round_index, finalRound?.task_id)
          ),
        }
      : result.final_round_result,
    manual_report: finalRound?.manual_report || result.manual_report,
    continue_mining_request: finalRound?.continue_mining_request || result.continue_mining_request,
  };
};

const collectRDAgentResultExpressions = (
  result?: AutoCampaignResult | AutoCampaignStatus | null,
): string[] => {
  const seen = new Set<string>();
  const expressions: string[] = [];
  const addExpression = (value: unknown) => {
    const expression = String(value || "").trim();
    const key = expression.toLowerCase().replace(/\s+/g, "");
    if (!expression || seen.has(key)) return;
    seen.add(key);
    expressions.push(expression);
  };
  const addFactors = (factors?: any[]) => {
    (factors || []).forEach((factor) => addExpression(factor?.expression));
  };

  addFactors((result as AutoCampaignResult | undefined)?.retained_factors);
  addFactors((result as AutoCampaignStatus | undefined)?.candidates);
  (result?.rounds || []).forEach((round: any) => {
    addFactors(round?.candidates || round?.all_factors);
    addFactors(round?.retained_factors);
  });
  addFactors((result as AutoCampaignResult | undefined)?.final_round_result?.factors);

  return expressions;
};

const buildContinuationInsightSummary = (
  autoResult: AutoMiningResult | null,
  autoStatus: MiningStatus | null,
): ContinuationInsightSummary | null => {
  const bestFactor = autoResult?.factors?.[0] || autoResult?.parent_factor;
  const details = (bestFactor?.task_details || bestFactor?.quantgpt_task_details || {}) as Record<string, any>;
  const roundEvaluation = (
    autoResult?.round_evaluation
    || details.round_evaluation
    || null
  ) as Record<string, any> | null;
  const interpretation = (details.interpretation || bestFactor?.interpretation || {}) as Record<string, any>;
  const params = (details.params || {}) as Record<string, any>;
  const baseFactors = Array.isArray(roundEvaluation?.base_factors)
    ? roundEvaluation?.base_factors.filter(Boolean)
    : (Array.isArray(params.base_factors) ? params.base_factors.filter(Boolean) : []);

  const weaknessCandidates: string[] = [];
  const directionCandidates: string[] = [];

  const pushStrings = (target: string[], value: any) => {
    if (Array.isArray(value)) {
      value.forEach((item) => {
        const text = String(item || "").trim();
        if (text) target.push(text);
      });
      return;
    }
    const text = String(value || "").trim();
    if (text) target.push(text);
  };

  pushStrings(weaknessCandidates, interpretation.weaknesses);
  pushStrings(weaknessCandidates, interpretation.risks);
  pushStrings(weaknessCandidates, interpretation.limitations);
  pushStrings(weaknessCandidates, interpretation.risk);
  pushStrings(weaknessCandidates, interpretation.summary);
  pushStrings(weaknessCandidates, roundEvaluation?.primary_problem);
  pushStrings(weaknessCandidates, roundEvaluation?.secondary_problem);

  pushStrings(directionCandidates, interpretation.next_steps);
  pushStrings(directionCandidates, interpretation.improvement_ideas);
  pushStrings(directionCandidates, interpretation.explanation);
  pushStrings(directionCandidates, roundEvaluation?.recommended_goal);
  pushStrings(directionCandidates, roundEvaluation?.suggested_actions);

  const dedupe = (items: string[]) => Array.from(new Set(items.map((item) => item.trim()).filter(Boolean)));
  const weaknesses = dedupe(weaknessCandidates);
  const optimizationDirections = dedupe(directionCandidates);

  if (!baseFactors.length && !weaknesses.length && !optimizationDirections.length && !autoStatus?.task_id) {
    return null;
  }

  return {
    baseFactors,
    weaknesses,
    optimizationDirections,
  };
};

const getRoundEvaluation = (result: AutoMiningResult | null): Record<string, any> | null => {
  if (!result) return null;
  const bestFactor = result.factors?.[0] || result.parent_factor;
  return (
    result.round_evaluation
    || bestFactor?.task_details?.round_evaluation
    || bestFactor?.quantgpt_task_details?.round_evaluation
    || null
  );
};

const loadPersistedRDAgentTaskState = (): PersistedRDAgentTaskState | null => {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(RDAGENT_ACTIVE_TASK_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed?.taskId || typeof parsed.taskId !== "string") return null;
    return {
      taskId: parsed.taskId,
      status: typeof parsed.status === "string" ? parsed.status : undefined,
      updatedAt: typeof parsed.updatedAt === "string" ? parsed.updatedAt : undefined,
    };
  } catch (error) {
    console.warn("读取 RDAgent 后台任务状态失败:", error);
    return null;
  }
};

const persistRDAgentTaskState = (taskId: string, status?: string) => {
  if (typeof window === "undefined" || !taskId) return;
  try {
    window.localStorage.setItem(
      RDAGENT_ACTIVE_TASK_STORAGE_KEY,
      JSON.stringify({
        taskId,
        status: status || "running",
        updatedAt: new Date().toISOString(),
      }),
    );
  } catch (error) {
    console.warn("保存 RDAgent 后台任务状态失败:", error);
  }
};

const clearPersistedRDAgentTaskState = () => {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(RDAGENT_ACTIVE_TASK_STORAGE_KEY);
  } catch (error) {
    console.warn("清理 RDAgent 后台任务状态失败:", error);
  }
};

const isRDAgentTerminalStatus = (status?: string | null) =>
  RDAGENT_TERMINAL_STATUSES.includes(String(status || "").toLowerCase() as (typeof RDAGENT_TERMINAL_STATUSES)[number]);

const isRDAgentResumableStatus = (status?: string | null) =>
  RDAGENT_RESUMABLE_STATUSES.includes(String(status || "").toLowerCase() as (typeof RDAGENT_RESUMABLE_STATUSES)[number]);

const ResultSection: React.FC<{
  kicker?: string;
  title: string;
  description?: string;
  extra?: React.ReactNode;
  children: React.ReactNode;
}> = ({ kicker, title, description, extra, children }) => (
  <section className="result-section-card">
    <div className="result-section-header">
      <div>
        {kicker ? <p className="result-section-kicker">{kicker}</p> : null}
        <h3 className="result-title" style={{ marginBottom: description ? 8 : 0 }}>{title}</h3>
        {description ? <p className="result-section-copy">{description}</p> : null}
      </div>
      {extra ? <div className="result-section-actions">{extra}</div> : null}
    </div>
    {children}
  </section>
);

const FactorMining: React.FC = () => {
  const [manualForm] = Form.useForm();
  const [autoForm] = Form.useForm();
  const [rdAgentForm] = Form.useForm();
  const [llmForm] = Form.useForm();
  const [continueForm] = Form.useForm();
  const watchedManualBaseFactors = Form.useWatch("base_factors", manualForm) as string[] | undefined;
  const watchedAutoBaseFactors = Form.useWatch("base_factors", autoForm) as string[] | undefined;

  const progressChartRef = useRef<HTMLDivElement>(null);
  const resultChartRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);
  const resultChartInstanceRef = useRef<echarts.ECharts | null>(null);
  const pollRef = useRef<number | null>(null);
  const elapsedRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number | null>(null);
  const restoringRDAgentTaskRef = useRef<string | null>(null);

  const [activeTab, setActiveTab] = useState<MiningMode>("manual");
  const [factors, setFactors] = useState<Factor[]>([]);
  const [loading, setLoading] = useState(false);
  const [mining, setMining] = useState(false);
  const [currentStockCode, setCurrentStockCode] = useState("");
  const [elapsedTime, setElapsedTime] = useState(0);
  const [, setSavedFactorNames] = useState<Set<string>>(new Set());

  const [manualStatus, setManualStatus] = useState<MiningStatus | null>(null);
  const [manualResult, setManualResult] = useState<ManualMiningResult | null>(null);
  const [autoStatus, setAutoStatus] = useState<MiningStatus | null>(null);
  const [autoResult, setAutoResult] = useState<AutoMiningResult | null>(null);
  const [autoCampaignStatus, setAutoCampaignStatus] = useState<AutoCampaignStatus | null>(null);
  const [autoCampaignResult, setAutoCampaignResult] = useState<AutoCampaignResult | null>(null);
  const [llmConfig, setLlmConfig] = useState<any>(null);
  const [llmConfigLoading, setLlmConfigLoading] = useState(false);
  const [llmConfigSaving, setLlmConfigSaving] = useState(false);
  const [llmRestarting, setLlmRestarting] = useState(false);
  const [llmSelectingFactors, setLlmSelectingFactors] = useState(false);
  const [continueExploring, setContinueExploring] = useState(false);
  const [rdagentContinuing, setRdagentContinuing] = useState(false);
  const [rdagentLoadingLatest, setRdagentLoadingLatest] = useState(false);
  const [continueSelectingFactors, setContinueSelectingFactors] = useState(false);
  const [showContinuePanel, setShowContinuePanel] = useState(false);
  const [llmSelectionSummary, setLlmSelectionSummary] = useState<string>("");
  const [rdagentSelectionSummary, setRdagentSelectionSummary] = useState<string>("");
  const [continueSelectionSummary, setContinueSelectionSummary] = useState<string>("");
  const [rdagentBootstrapMode, setRdagentBootstrapMode] = useState<RDAgentBootstrapMode>("llm_auto");
  const [expandedReportUrl, setExpandedReportUrl] = useState<string | null>(null);
  const [expandedDetailsKey, setExpandedDetailsKey] = useState<string | null>(null);
  const [isMobileView, setIsMobileView] = useState(typeof window !== "undefined" ? window.innerWidth <= 768 : false);
  const [autoSeedState, setAutoSeedState] = useState<AutoMiningSeedState | null>(null);
  const [autoWorkflowMode, setAutoWorkflowMode] = useState<"single" | "campaign">("single");
  const [autoLaunchMode, setAutoLaunchMode] = useState<"single" | "campaign">("campaign");
  const [persistedRDAgentTaskState, setPersistedRDAgentTaskState] = useState<PersistedRDAgentTaskState | null>(() =>
    loadPersistedRDAgentTaskState()
  );

  const currentStatus = activeTab === "manual" ? manualStatus : autoStatus;
  const continuationInsightSummary = useMemo(
    () => buildContinuationInsightSummary(autoResult, autoStatus),
    [autoResult, autoStatus]
  );
  const currentResult = activeTab === "manual" ? manualResult : autoResult;
  const isAutoLikeTab = activeTab !== "manual";

  const getRDAgentTotalCandidates = (values: Record<string, any>) =>
    Number(values.max_iterations || 1) * Number(values.candidates_per_iteration || 1);

  const loadFactors = async () => {
    try {
      const response = (await api.getFactors()) as any;
      if (response.success) setFactors(response.data || []);
    } catch (error) {
      console.error("加载因子列表失败:", error);
    }
  };

  const loadLLMConfig = async () => {
    setLlmConfigLoading(true);
    try {
      const response = (await api.getLLMConfig()) as any;
      setLlmConfig(response);
      llmForm.setFieldsValue({
        api_key: "",
        base_url: response?.base_url || "https://api.deepseek.com/v1",
        model: response?.model || "deepseek-chat",
      });
    } catch (error) {
      console.error("加载 LLM 配置失败:", error);
    } finally {
      setLlmConfigLoading(false);
    }
  };

  const clearTimers = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    if (elapsedRef.current) {
      clearInterval(elapsedRef.current);
      elapsedRef.current = null;
    }
    startTimeRef.current = null;
  };

  const syncPersistedRDAgentTaskState = (taskId: string, status?: string) => {
    persistRDAgentTaskState(taskId, status);
    setPersistedRDAgentTaskState({
      taskId,
      status: status || "running",
      updatedAt: new Date().toISOString(),
    });
  };

  const clearPersistedRDAgentTaskStateAndBanner = () => {
    clearPersistedRDAgentTaskState();
    setPersistedRDAgentTaskState(null);
  };

  useEffect(() => {
    const handleResize = () => setIsMobileView(window.innerWidth <= 768);
    if (typeof window !== "undefined") {
      window.addEventListener("resize", handleResize);
    }

    loadFactors();
    loadLLMConfig();
    const endDate = dayjs();
    const startDate = dayjs().subtract(1, "year");

    manualForm.setFieldsValue({
      stock_code: "000001",
      dateRange: [startDate, endDate],
      manual_prompt: "为当前股票挑选一组更适合手动遗传挖掘的基础因子，优先保证稳定性和可解释性。",
      manual_direction: "stability",
      manual_max_factor_count: 8,
      population_size: 50,
      n_generations: 10,
      mutation_rate: 0.2,
      crossover_rate: 0.7,
      elite_size: 5,
      fitness_objective: "ic_mean",
      ic_threshold: 0.03,
    });

    const defaultAutoValues = {
      prompt: "基于已导入因子做自动因子挖掘，优先生成可解释、稳定的量价复合因子",
      dateRange: [startDate, endDate],
      universe: "hs300",
      benchmark: "hs300",
      n_groups: 5,
      holding_period: 5,
      n_candidates: 5,
      max_factor_count: 12,
      neutralize_industry: true,
      neutralize_cap: true,
    };

    let persistedAutoValues: Record<string, any> = {};
    if (typeof window !== "undefined") {
      try {
        const raw = window.localStorage.getItem(AUTO_MINING_FORM_STORAGE_KEY);
        if (raw) {
          const parsed = JSON.parse(raw);
          persistedAutoValues = {
            ...parsed,
            dateRange:
              Array.isArray(parsed?.dateRange) && parsed.dateRange.length === 2
                ? [dayjs(parsed.dateRange[0]), dayjs(parsed.dateRange[1])]
                : undefined,
          };
        }
      } catch (error) {
        console.warn("读取自动挖掘本地配置失败:", error);
      }
    }

    autoForm.setFieldsValue({
      ...defaultAutoValues,
      ...persistedAutoValues,
      automation_exploration_rounds: 3,
      automation_n_candidates_per_round: 5,
      automation_additional_factor_count_per_round: 3,
      automation_factor_update_mode: "append",
      automation_parent_selection_strategy: "best_score_so_far",
      automation_match_mode: "all",
      automation_wq_ratings: [],
    });
    if (persistedAutoValues?.preferred_launch_mode === "single" || persistedAutoValues?.preferred_launch_mode === "campaign") {
      setAutoLaunchMode(persistedAutoValues.preferred_launch_mode);
    }

    rdAgentForm.setFieldsValue({
      objective: "基于 FactorHub 因子库持续生成、回测并筛选新的 SOTA 候选因子",
      dateRange: [startDate, endDate],
      candidate_universe: ["close", "volume", "amount"],
      base_factors: [],
      universe: "hs300",
      benchmark: "hs300",
      max_iterations: 6,
      candidates_per_iteration: 3,
      n_groups: 5,
      holding_period: 5,
      direction: "score",
      neutralize_industry: true,
      neutralize_cap: true,
      sota_library_id: "factorhub-sota",
      max_correlation_with_sota: 0.99,
      min_rank_ic: 0,
      min_annualized_return_delta: 0,
      max_drawdown_regression: 0.05,
      min_valid_coverage: 0.8,
    });

    const persistedRDAgentTask = loadPersistedRDAgentTaskState();
    if (persistedRDAgentTask?.taskId && isRDAgentResumableStatus(persistedRDAgentTask.status)) {
      setPersistedRDAgentTaskState(persistedRDAgentTask);
      void restoreRDAgentTask(persistedRDAgentTask.taskId, true);
    } else if (persistedRDAgentTask?.taskId && isRDAgentTerminalStatus(persistedRDAgentTask.status)) {
      clearPersistedRDAgentTaskStateAndBanner();
    }

    return () => {
      if (typeof window !== "undefined") {
        window.removeEventListener("resize", handleResize);
      }
      clearTimers();
      chartRef.current?.dispose();
      resultChartInstanceRef.current?.dispose();
    };
  }, []);

  useEffect(() => {
    if (typeof document === "undefined" || typeof window === "undefined") {
      return;
    }

    const maybeReconnect = () => {
      if (document.visibilityState !== "visible") return;
      const latestPersisted = loadPersistedRDAgentTaskState();
      if (!latestPersisted?.taskId) return;
      if (isRDAgentTerminalStatus(latestPersisted.status)) {
        clearPersistedRDAgentTaskStateAndBanner();
        return;
      }
      setPersistedRDAgentTaskState(latestPersisted);
      const shouldReconnect = isRDAgentResumableStatus(latestPersisted.status);
      if (!shouldReconnect) return;
      if (mining) return;
      if (restoringRDAgentTaskRef.current === latestPersisted.taskId) return;
      void restoreRDAgentTask(latestPersisted.taskId, true);
    };

    document.addEventListener("visibilitychange", maybeReconnect);
    window.addEventListener("focus", maybeReconnect);

    return () => {
      document.removeEventListener("visibilitychange", maybeReconnect);
      window.removeEventListener("focus", maybeReconnect);
    };
  }, [activeTab, mining]);

  useEffect(() => {
    if (activeTab === "manual" && manualResult?.fitness_history) {
      if (typeof window !== "undefined") {
        const timer = window.setTimeout(() => {
          updateChart("result", manualResult.fitness_history, "完整进化曲线", "适应度");
        }, 200);
        return () => window.clearTimeout(timer);
      }
      updateChart("result", manualResult.fitness_history, "完整进化曲线", "适应度");
    }
  }, [activeTab, manualResult]);

  useEffect(() => {
    if (activeTab === "auto" && autoResult?.fitness_history) {
      if (typeof window !== "undefined") {
        window.requestAnimationFrame(() => {
          updateChart("result", autoResult.fitness_history, "完整研究曲线", "综合分数");
        });
      } else {
        updateChart("result", autoResult.fitness_history, "完整研究曲线", "综合分数");
      }
    }
  }, [activeTab, autoResult]);

  useEffect(() => {
    if (isAutoLikeTab && autoWorkflowMode === "campaign" && autoCampaignResult?.fitness_history) {
      if (typeof window !== "undefined") {
        window.requestAnimationFrame(() => {
          updateChart("result", autoCampaignResult.fitness_history, activeTab === "rdagent" ? "RDAgent 研究曲线" : "自动化完整研究曲线", "综合分数");
        });
      } else {
        updateChart("result", autoCampaignResult.fitness_history, activeTab === "rdagent" ? "RDAgent 研究曲线" : "自动化完整研究曲线", "综合分数");
      }
    }
  }, [activeTab, autoCampaignResult, autoWorkflowMode, isAutoLikeTab]);

  useEffect(() => {
    const seedFitnessHistory = autoSeedState?.result?.fitness_history;
    if (activeTab === "auto" && seedFitnessHistory) {
      if (typeof window !== "undefined") {
        window.requestAnimationFrame(() => {
          updateChart("result", seedFitnessHistory, "上一轮完整研究曲线", "综合分数");
        });
      } else {
        updateChart("result", seedFitnessHistory, "上一轮完整研究曲线", "综合分数");
      }
    }
  }, [activeTab, autoSeedState]);

  useEffect(() => {
    const chartHistory = activeTab === "manual" && mining ? buildManualChartHistory(manualStatus) : undefined;
    if (chartHistory) {
      if (typeof window !== "undefined") {
        window.requestAnimationFrame(() => {
          updateChart("progress", chartHistory, "进化曲线", "适应度");
        });
      } else {
        updateChart("progress", chartHistory, "进化曲线", "适应度");
      }
    }
  }, [activeTab, mining, manualStatus]);

  useEffect(() => {
    if (activeTab === "auto" && autoWorkflowMode === "single" && mining && autoStatus?.fitness_history) {
      if (typeof window !== "undefined") {
        window.requestAnimationFrame(() => {
          updateChart("progress", autoStatus.fitness_history, "研究曲线", "综合分数");
        });
      } else {
        updateChart("progress", autoStatus.fitness_history, "研究曲线", "综合分数");
      }
    }
  }, [activeTab, autoWorkflowMode, mining, autoStatus]);

  useEffect(() => {
    if (isAutoLikeTab && autoWorkflowMode === "campaign" && mining && autoCampaignStatus?.fitness_history) {
      if (typeof window !== "undefined") {
        window.requestAnimationFrame(() => {
          updateChart("progress", autoCampaignStatus.fitness_history, activeTab === "rdagent" ? "RDAgent 研究曲线" : "自动化研究曲线", "综合分数");
        });
      } else {
        updateChart("progress", autoCampaignStatus.fitness_history, activeTab === "rdagent" ? "RDAgent 研究曲线" : "自动化研究曲线", "综合分数");
      }
    }
  }, [activeTab, autoWorkflowMode, mining, autoCampaignStatus, isAutoLikeTab]);

  const updateChart = (
    target: "progress" | "result",
    fitnessHistory?: { best: number[]; average: number[] },
    title?: string,
    yName?: string,
  ) => {
    if (!fitnessHistory) return;
    if (!(fitnessHistory.best?.length || fitnessHistory.average?.length)) return;
    const ref = target === "progress" ? progressChartRef.current : resultChartRef.current;
    if (!ref) return;

    let chart = target === "progress" ? chartRef.current : resultChartInstanceRef.current;
    if (chart && chart.getDom() !== ref) {
      chart.dispose();
      if (target === "progress") chartRef.current = null;
      else resultChartInstanceRef.current = null;
      chart = null;
    }

    if (!chart) {
      chart = echarts.init(ref);
      if (target === "progress") chartRef.current = chart;
      else resultChartInstanceRef.current = chart;
    }

    const generations = fitnessHistory.best.map((_, i) => i + 1);
    chart.setOption(
      {
        title: { text: title || "进化曲线", left: "center", textStyle: { fontSize: 16, fontWeight: 600 } },
        tooltip: { trigger: "axis" },
        legend: { data: ["最优值", "平均值"], bottom: 0 },
        grid: { left: "3%", right: "4%", bottom: "10%", containLabel: true },
        xAxis: { type: "category", name: "轮次", data: generations },
        yAxis: { type: "value", name: yName || "分数", scale: true },
        series: [
          { name: "最优值", type: "line", data: fitnessHistory.best, smooth: true, itemStyle: { color: "#3b82f6" } },
          { name: "平均值", type: "line", data: fitnessHistory.average, smooth: true, itemStyle: { color: "#22c55e" } },
        ],
      },
      true,
    );
    chart.resize();
  };

  const startClock = () => {
    setElapsedTime(0);
    startTimeRef.current = Date.now();
    elapsedRef.current = setInterval(() => {
      if (!startTimeRef.current) return;
      setElapsedTime(Math.floor((Date.now() - startTimeRef.current) / 1000));
    }, 1000);
  };

  const formatElapsedTime = (seconds: number) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    if (h > 0) return `${h}小时${m}分${s}秒`;
    if (m > 0) return `${m}分${s}秒`;
    return `${s}秒`;
  };

  const restoreRDAgentTask = async (taskId: string, silent = false) => {
    if (!taskId) return;
    if (restoringRDAgentTaskRef.current === taskId) return;
    try {
      restoringRDAgentTaskRef.current = taskId;
      setRdagentLoadingLatest(true);
      setActiveTab("rdagent");
      setAutoWorkflowMode("campaign");
      const statusResponse = (await api.getRDAgentMiningStatus(taskId)) as any;
      if (!statusResponse?.success) {
        throw new Error("读取 RDAgent 任务状态失败");
      }

      const rawStatus = statusResponse.data as AutoCampaignStatus;
      const status = {
        ...rawStatus,
        candidates: (rawStatus.candidates || []).map((factor: any) =>
          normalizeRDAgentFactorForDashboard(factor, rawStatus.current_round, taskId)
        ),
        rounds: (rawStatus.rounds || []).map((round: any) => {
          const factors = (round.candidates || round.all_factors || []).map((factor: any) =>
            normalizeRDAgentFactorForDashboard(factor, round.round_index, round.task_id || taskId)
          );
          return { ...round, candidates: factors, all_factors: factors };
        }),
        latest_round: rawStatus.latest_round
          ? {
              ...rawStatus.latest_round,
              candidates: (rawStatus.latest_round.candidates || rawStatus.latest_round.all_factors || []).map((factor: any) =>
                normalizeRDAgentFactorForDashboard(factor, rawStatus.latest_round?.round_index, rawStatus.latest_round?.task_id || taskId)
              ),
            }
          : rawStatus.latest_round,
      } as AutoCampaignStatus;

      setAutoCampaignStatus(status);
      setAutoCampaignResult(null);
      setCurrentStockCode("RDAGENT");

      if (status.status === "completed") {
        const resultResponse = (await api.getRDAgentMiningResults(taskId)) as any;
        if (!resultResponse?.success) {
          throw new Error("读取 RDAgent 任务结果失败");
        }
        const normalizedResult = normalizeRDAgentResultForDashboard({
          ...resultResponse.data,
          fitness_history: resultResponse.data?.fitness_history || status.fitness_history || { best: [], average: [] },
        });
        setAutoCampaignResult(normalizedResult);
        setMining(false);
        clearTimers();
        clearPersistedRDAgentTaskStateAndBanner();
        updateChart("result", normalizedResult?.fitness_history || status.fitness_history, "RDAgent 完整研究曲线", "综合分数");
        if (!silent) message.success("已恢复最近一次 RDAgent 挖掘结果");
        return;
      }

      if (status.status === "failed") {
        setMining(false);
        clearTimers();
        clearPersistedRDAgentTaskStateAndBanner();
        if (!silent) message.warning(`已恢复失败任务状态：${status.error || "未知错误"}`);
        return;
      }

      if (status.status === "cancelled") {
        setMining(false);
        clearTimers();
        clearPersistedRDAgentTaskStateAndBanner();
        if (!silent) message.info(status.error || "RDAgent 任务已终止");
        return;
      }

      setMining(true);
      clearTimers();
      startClock();
      updateChart("progress", status.fitness_history, "RDAgent 研究曲线", "综合分数");
      pollRef.current = window.setInterval(() => checkRDAgentProgress(taskId), AUTO_PROGRESS_POLL_MS);
      syncPersistedRDAgentTaskState(taskId, status.status);
      if (!silent) message.success("已重新连接到后台 RDAgent 挖掘任务");
    } catch (error: any) {
      clearPersistedRDAgentTaskStateAndBanner();
      if (!silent) {
        message.error(error?.message || "恢复 RDAgent 后台任务失败");
      }
    } finally {
      restoringRDAgentTaskRef.current = null;
      setRdagentLoadingLatest(false);
    }
  };

  const startManualMining = async (values: any) => {
    const [startDate, endDate] = values.dateRange;
    const requestData = {
      stock_code: values.stock_code,
      base_factors: values.base_factors || [],
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
      setManualResult(null);
      setManualStatus({
        task_id: "pending",
        status: "pending",
        current_generation: 0,
        total_generations: Number(values.n_generations || 0),
        best_fitness: 0,
        avg_fitness: 0,
        fitness_history: { best: [], average: [] },
        candidates: [],
      });
      setSavedFactorNames(new Set());
      setCurrentStockCode(String(values.stock_code || "").replace(".", ""));
      clearTimers();
      startClock();
      const response = (await api.startGeneticMining(requestData)) as any;
      const taskId = response?.data?.task_id;
      if (!taskId) throw new Error("未返回任务ID");
      pollRef.current = window.setInterval(() => checkManualProgress(taskId), 2000);
      message.success("手动挖掘任务已启动");
    } catch (error) {
      console.error(error);
      clearTimers();
      setMining(false);
      message.error("启动手动挖掘失败");
    } finally {
      setLoading(false);
    }
  };

  const checkManualProgress = async (taskId: string) => {
    try {
      const response = (await api.getMiningStatus(taskId)) as any;
      if (!response.success) return;
      const rawStatus = response.data as MiningStatus;
      const chartHistory = buildManualChartHistory(rawStatus);
      const status = {
        ...rawStatus,
        fitness_history: chartHistory || rawStatus.fitness_history || { best: [], average: [] },
      } as MiningStatus;
      setManualStatus(status);
      if (chartHistory) {
        if (typeof window !== "undefined") {
          window.requestAnimationFrame(() => {
            updateChart("progress", chartHistory, "进化曲线", "适应度");
          });
        } else {
          updateChart("progress", chartHistory, "进化曲线", "适应度");
        }
      }
      if (status.status === "completed") {
        clearTimers();
        setMining(false);
        const result = (await api.getMiningResults(taskId)) as any;
        if (result.success) {
          const resultFitnessHistory = result.data?.fitness_history || chartHistory || status.fitness_history || { best: [], average: [] };
          setManualResult({
            ...result.data,
            fitness_history: resultFitnessHistory,
          });
        }
      } else if (status.status === "failed") {
        clearTimers();
        setMining(false);
        message.error(`手动挖掘失败: ${status.error || "未知错误"}`);
      }
    } catch (error) {
      console.error(error);
    }
  };

  const saveLLMConfig = async () => {
    try {
      const values = await llmForm.validateFields();
      setLlmConfigSaving(true);
      const response = (await api.saveLLMConfig({
        api_key: values.api_key ? values.api_key : null,
        base_url: values.base_url,
        model: values.model,
      })) as any;
      setLlmConfig(response);
      llmForm.setFieldsValue({ api_key: "" });
      message.success(response?.message || "LLM 配置已保存");
    } catch (error: any) {
      message.error(error?.message || "保存 LLM 配置失败");
    } finally {
      setLlmConfigSaving(false);
    }
  };

  const restartLLMService = async () => {
    try {
      setLlmRestarting(true);
      const response = (await api.restartLLMService()) as any;
      message.success(response?.message || 'LLM 服务状态检查完成');
      await loadLLMConfig();
    } catch (error: any) {
      message.error(error?.message || '检查 LLM 服务状态失败');
    } finally {
      setLlmRestarting(false);
    }
  };

  const handleLLMSelectFactors = async () => {
    try {
      const values = await autoForm.validateFields(["prompt", "dateRange", "universe", "benchmark", "direction", "max_factor_count"]);
      const [startDate, endDate] = values.dateRange;
      setLlmSelectingFactors(true);
      setLlmSelectionSummary("");
      const response = (await autoMiningApi.selectFactors({
        prompt: values.prompt,
        direction: values.direction,
        start_date: startDate.format("YYYY-MM-DD"),
        end_date: endDate.format("YYYY-MM-DD"),
        universe: values.universe,
        benchmark: values.benchmark,
        max_factor_count: Number(values.max_factor_count || 12),
        candidate_limit: 80,
      })) as any;
      const selectedFactors = response?.data?.selected_factors || [];
      if (!selectedFactors.length) {
        throw new Error("LLM 未返回可用因子");
      }
      const nextValues = { ...autoForm.getFieldsValue(), base_factors: selectedFactors };
      autoForm.setFieldsValue({ base_factors: selectedFactors });
      persistAutoMiningForm({}, nextValues);
      setLlmSelectionSummary(response?.data?.selection_rationale || "");
      message.success(response?.message || `LLM 已在上限范围内自动筛选 ${selectedFactors.length} 个基础因子`);
    } catch (error: any) {
      message.error(error?.message || "LLM 自动筛选因子失败");
    } finally {
      setLlmSelectingFactors(false);
    }
  };

  const handleManualLLMSelectFactors = async () => {
    try {
      const values = await manualForm.validateFields(["manual_prompt", "dateRange", "manual_direction", "manual_max_factor_count"]);
      const [startDate, endDate] = values.dateRange;
      const manualPrompt = String(values.manual_prompt || "").trim() || "为当前股票挑选一组更适合手动遗传挖掘的基础因子，优先保证稳定性和可解释性。";
      setLlmSelectingFactors(true);
      setLlmSelectionSummary("");
      const response = (await autoMiningApi.selectFactors({
        prompt: manualPrompt,
        direction: values.manual_direction,
        start_date: startDate.format("YYYY-MM-DD"),
        end_date: endDate.format("YYYY-MM-DD"),
        universe: "single_stock",
        benchmark: "hs300",
        max_factor_count: Number(values.manual_max_factor_count || 8),
        candidate_limit: 80,
        selection_mode: "manual_genetic",
      })) as any;
      const selectedFactors = response?.data?.selected_factors || [];
      if (!selectedFactors.length) {
        throw new Error("LLM 未返回可用因子");
      }
      manualForm.setFieldsValue({ base_factors: selectedFactors });
      setLlmSelectionSummary(response?.data?.selection_rationale || "");
      message.success(response?.message || `LLM 已自动筛选 ${selectedFactors.length} 个手动挖掘基础因子`);
    } catch (error: any) {
      message.error(error?.message || "LLM 自动筛选手动挖掘因子失败");
    } finally {
      setLlmSelectingFactors(false);
    }
  };

  const handleRDAgentLLMBootstrap = async () => {
    try {
      const values = await rdAgentForm.validateFields(["objective", "dateRange", "universe", "benchmark", "direction"]);
      const [startDate, endDate] = values.dateRange;
      setLlmSelectingFactors(true);
      setRdagentSelectionSummary("");
      const response = (await api.selectRDAgentBootstrap({
        objective: values.objective,
        direction: values.direction,
        start_date: startDate.format("YYYY-MM-DD"),
        end_date: endDate.format("YYYY-MM-DD"),
        universe: values.universe,
        benchmark: values.benchmark,
        max_factor_count: 8,
        max_candidate_field_count: 5,
        candidate_limit: 80,
      })) as any;
      const candidateUniverse = response?.data?.candidate_universe || [];
      const baseFactors = response?.data?.base_factors || [];
      if (!candidateUniverse.length) {
        throw new Error("LLM 未返回可用的候选字段");
      }
      if (!baseFactors.length) {
        throw new Error("LLM 未返回可用的基础因子");
      }
      rdAgentForm.setFieldsValue({
        candidate_universe: candidateUniverse,
        base_factors: baseFactors,
      });
      const fieldReason = response?.data?.field_reason || {};
      const fieldSummary = candidateUniverse
        .map((field: string) => `${field}${fieldReason?.[field] ? `：${fieldReason[field]}` : ""}`)
        .join("；");
      const rationale = String(response?.data?.selection_rationale || "").trim();
      setRdagentSelectionSummary(
        [rationale, fieldSummary ? `字段分工：${fieldSummary}` : ""].filter(Boolean).join("\n\n")
      );
      message.success(
        response?.message ||
        `LLM 已为 RDAgent 自动选择 ${candidateUniverse.length} 个候选字段和 ${baseFactors.length} 个基础因子`
      );
    } catch (error: any) {
      message.error(error?.message || "RDAgent 自动生成启动配置失败");
    } finally {
      setLlmSelectingFactors(false);
    }
  };

  const renderRDAgentBootstrapSummary = () => {
    if (!rdagentSelectionSummary) return null;
    return (
      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16, whiteSpace: "pre-wrap" }}
        message="RDAgent 启动配置说明"
        description={rdagentSelectionSummary}
      />
    );
  };

  const startAutoMining = async (values: any) => {
    persistAutoMiningForm({}, values);
    const [startDate, endDate] = values.dateRange;
    const requestData = {
      prompt: values.prompt,
      base_factors: values.base_factors || [],
      start_date: startDate.format("YYYY-MM-DD"),
      end_date: endDate.format("YYYY-MM-DD"),
      universe: values.universe,
      benchmark: values.benchmark,
      n_groups: values.n_groups,
      holding_period: values.holding_period,
      n_candidates: values.n_candidates,
      direction: values.direction,
      neutralize_industry: values.neutralize_industry ?? true,
      neutralize_cap: values.neutralize_cap ?? true,
    };

    try {
      setLoading(true);
      setMining(true);
      setAutoWorkflowMode("single");
      setAutoSeedState(null);
      setAutoCampaignStatus(null);
      setAutoCampaignResult(null);
      setAutoResult(null);
      setAutoStatus({
        task_id: "pending",
        status: "pending",
        current_generation: 0,
        total_generations: Number(values.n_candidates || 0),
        best_fitness: 0,
        avg_fitness: 0,
        fitness_history: { best: [], average: [] },
      });
      setSavedFactorNames(new Set());
      setCurrentStockCode(values.universe || "AUTO");
      clearTimers();
      startClock();
      const response = (await autoMiningApi.startTask(requestData)) as any;
      const taskId = response?.data?.task_id;
      if (!taskId) throw new Error("未返回任务ID");
      await checkAutoProgress(taskId);
      pollRef.current = window.setInterval(() => checkAutoProgress(taskId), AUTO_PROGRESS_POLL_MS);
      message.success("自动挖掘任务已启动");
    } catch (error) {
      console.error(error);
      clearTimers();
      setMining(false);
      message.error("启动自动挖掘失败");
    } finally {
      setLoading(false);
    }
  };

  const startAutoMiningCampaign = async () => {
    try {
      const autoValues = await autoForm.validateFields([
        "prompt",
        "base_factors",
        "dateRange",
        "universe",
        "benchmark",
        "n_groups",
        "holding_period",
        "direction",
        "neutralize_industry",
        "neutralize_cap",
      ]);
      const automationValues = await autoForm.validateFields([
        "automation_exploration_rounds",
        "automation_n_candidates_per_round",
        "automation_additional_factor_count_per_round",
        "automation_factor_update_mode",
        "automation_parent_selection_strategy",
        "automation_match_mode",
        "automation_score_min",
        "automation_wq_ratings",
        "automation_ls_sharpe_min",
        "automation_ls_return_min",
        "automation_wq_return_min",
      ]);
      const [startDate, endDate] = autoValues.dateRange;

      setLoading(true);
      setMining(true);
      setAutoWorkflowMode("campaign");
      setAutoSeedState(null);
      setAutoResult(null);
      setAutoStatus(null);
      setAutoCampaignResult(null);
      setAutoCampaignStatus({
        task_id: "pending",
        status: "pending",
        progress: 0,
        current_round: 0,
        total_rounds: Number(automationValues.automation_exploration_rounds || 0),
        current_generation: 0,
        total_generations: Number(automationValues.automation_n_candidates_per_round || 0),
        best_fitness: 0,
        avg_fitness: 0,
        retained_count: 0,
        fitness_history: { best: [], average: [] },
      });
      clearTimers();
      startClock();

      const response = (await autoMiningApi.startCampaign({
        prompt: autoValues.prompt,
        base_factors: autoValues.base_factors || [],
        start_date: startDate.format("YYYY-MM-DD"),
        end_date: endDate.format("YYYY-MM-DD"),
        universe: autoValues.universe,
        benchmark: autoValues.benchmark,
        n_groups: Number(autoValues.n_groups || 5),
        holding_period: Number(autoValues.holding_period || 5),
        exploration_rounds: Number(automationValues.automation_exploration_rounds || 1),
        n_candidates_per_round: Number(automationValues.automation_n_candidates_per_round || 1),
        additional_factor_count_per_round: Number(automationValues.automation_additional_factor_count_per_round || 0),
        factor_update_mode: automationValues.automation_factor_update_mode || "append",
        parent_selection_strategy: automationValues.automation_parent_selection_strategy || "best_score_so_far",
        direction: autoValues.direction,
        neutralize_industry: autoValues.neutralize_industry ?? true,
        neutralize_cap: autoValues.neutralize_cap ?? true,
        retention_filter: {
          match_mode: automationValues.automation_match_mode || "all",
          score_min: automationValues.automation_score_min ?? undefined,
          wq_ratings: automationValues.automation_wq_ratings || [],
          ls_sharpe_min: automationValues.automation_ls_sharpe_min ?? undefined,
          ls_return_min: automationValues.automation_ls_return_min ?? undefined,
          wq_return_min: automationValues.automation_wq_return_min ?? undefined,
        },
      })) as any;

      const taskId = response?.data?.task_id;
      if (!taskId) throw new Error("未返回任务ID");
      await checkAutoCampaignProgress(taskId);
      pollRef.current = window.setInterval(() => checkAutoCampaignProgress(taskId), AUTO_PROGRESS_POLL_MS);
      message.success(response?.message || "全自动化挖掘任务已启动");
    } catch (error: any) {
      console.error(error);
      clearTimers();
      setMining(false);
      message.error(error?.message || "启动全自动化挖掘失败");
    } finally {
      setLoading(false);
    }
  };

  const handleAutoMiningSubmit = async (values: any) => {
    if (autoLaunchMode === "campaign") {
      await startAutoMiningCampaign();
      return;
    }
    await startAutoMining(values);
  };

  const getAutoSubmitButtonLabel = () => {
    if (mining) {
      return autoWorkflowMode === "campaign" ? "自动化挖掘中..." : "研究中...";
    }
    return autoLaunchMode === "campaign" ? "启动连续探索" : "开始单轮研究";
  };

  const checkAutoCampaignProgress = async (taskId: string) => {
    try {
      const response = (await autoMiningApi.getCampaignStatus(taskId)) as any;
      if (!response.success) return;
      const status = response.data as AutoCampaignStatus;
      setAutoCampaignStatus(status);
      updateChart("progress", status.fitness_history, "自动化研究曲线", "综合分数");
      if (status.status === "completed") {
        clearTimers();
        setMining(false);
        const result = (await autoMiningApi.getCampaignResult(taskId)) as any;
        if (result.success) {
          setAutoCampaignResult({
            ...result.data,
            fitness_history: result.data?.fitness_history || status.fitness_history || { best: [], average: [] },
          });
        }
      } else if (status.status === "failed") {
        clearTimers();
        setMining(false);
        message.error(`全自动化挖掘失败: ${status.error || "未知错误"}`);
      }
    } catch (error) {
      console.error(error);
    }
  };

  const startRDAgentMining = async (values: any) => {
    const [startDate, endDate] = values.dateRange;
    const previousExpressions = [
      ...collectRDAgentResultExpressions(autoCampaignResult),
      ...collectRDAgentResultExpressions(autoCampaignStatus),
    ];

    try {
      setLoading(true);
      setMining(true);
      setActiveTab("rdagent");
      setAutoWorkflowMode("campaign");
      setAutoSeedState(null);
      setAutoResult(null);
      setAutoStatus(null);
      setAutoCampaignResult(null);
      setAutoCampaignStatus({
        task_id: "pending",
        status: "pending",
        progress: 0,
        current_round: 0,
        total_rounds: Number(values.max_iterations || 0),
        current_generation: 0,
        total_generations: getRDAgentTotalCandidates(values),
        best_fitness: 0,
        avg_fitness: 0,
        retained_count: 0,
        fitness_history: { best: [], average: [] },
      });
      setSavedFactorNames(new Set());
      setCurrentStockCode(values.universe || "RDAGENT");
      clearTimers();
      startClock();

      const response = (await api.startRDAgentMining({
        objective: values.objective,
        candidate_universe: values.candidate_universe || ["close", "volume", "amount"],
        base_factors: values.base_factors || [],
        start_date: startDate.format("YYYY-MM-DD"),
        end_date: endDate.format("YYYY-MM-DD"),
        universe: values.universe,
        benchmark: values.benchmark,
        max_iterations: Number(values.max_iterations || 1),
        candidates_per_iteration: Number(values.candidates_per_iteration || 1),
        n_groups: Number(values.n_groups || 5),
        holding_period: Number(values.holding_period || 5),
        direction: values.direction,
        neutralize_industry: values.neutralize_industry ?? true,
        neutralize_cap: values.neutralize_cap ?? true,
        sota_library_id: values.sota_library_id || "factorhub-sota",
        previous_expressions: previousExpressions,
        acceptance_policy: {
          max_correlation_with_sota: Number(values.max_correlation_with_sota ?? 0.99),
          min_rank_ic: Number(values.min_rank_ic ?? 0),
          min_annualized_return_delta: Number(values.min_annualized_return_delta ?? 0),
          max_drawdown_regression: Number(values.max_drawdown_regression ?? 0.05),
          min_valid_coverage: Number(values.min_valid_coverage ?? 0.8),
        },
      })) as any;

      const taskId = response?.data?.task_id;
      if (!taskId) throw new Error("未返回任务ID");
      syncPersistedRDAgentTaskState(taskId, "pending");
      await checkRDAgentProgress(taskId);
      pollRef.current = window.setInterval(() => checkRDAgentProgress(taskId), AUTO_PROGRESS_POLL_MS);
      message.success(response?.message || "RDAgent 挖掘任务已启动");
    } catch (error: any) {
      console.error(error);
      clearTimers();
      setMining(false);
      message.error(error?.message || "启动 RDAgent 挖掘失败");
    } finally {
      setLoading(false);
    }
  };

  const checkRDAgentProgress = async (taskId: string) => {
    try {
      const response = (await api.getRDAgentMiningStatus(taskId)) as any;
      if (!response.success) return;
      const rawStatus = response.data as AutoCampaignStatus;
      const status = {
        ...rawStatus,
        candidates: (rawStatus.candidates || []).map((factor: any) =>
          normalizeRDAgentFactorForDashboard(factor, rawStatus.current_round, taskId)
        ),
        rounds: (rawStatus.rounds || []).map((round: any) => {
          const factors = (round.candidates || round.all_factors || []).map((factor: any) =>
            normalizeRDAgentFactorForDashboard(factor, round.round_index, round.task_id || taskId)
          );
          return { ...round, candidates: factors, all_factors: factors };
        }),
        latest_round: rawStatus.latest_round
          ? {
              ...rawStatus.latest_round,
              candidates: (rawStatus.latest_round.candidates || rawStatus.latest_round.all_factors || []).map((factor: any) =>
                normalizeRDAgentFactorForDashboard(factor, rawStatus.latest_round?.round_index, rawStatus.latest_round?.task_id || taskId)
              ),
            }
          : rawStatus.latest_round,
      } as AutoCampaignStatus;
      setAutoCampaignStatus(status);
      syncPersistedRDAgentTaskState(taskId, status.status);
      updateChart("progress", status.fitness_history, "RDAgent 研究曲线", "综合分数");
      if (status.status === "completed") {
        clearTimers();
        setMining(false);
        const result = (await api.getRDAgentMiningResults(taskId)) as any;
        if (result.success) {
          setAutoCampaignResult(normalizeRDAgentResultForDashboard({
            ...result.data,
            fitness_history: result.data?.fitness_history || status.fitness_history || { best: [], average: [] },
          }));
        }
      } else if (status.status === "failed") {
        clearTimers();
        setMining(false);
        message.error(`RDAgent 挖掘失败: ${status.error || "未知错误"}`);
      } else if (status.status === "cancelled") {
        clearTimers();
        setMining(false);
        clearPersistedRDAgentTaskStateAndBanner();
        message.info(status.error || "RDAgent 任务已终止");
      }
    } catch (error) {
      console.error(error);
    }
  };

  const loadLatestRDAgentResult = async () => {
    try {
      setRdagentLoadingLatest(true);
      setActiveTab("rdagent");
      setAutoWorkflowMode("campaign");
      const tasksResponse = (await api.listMiningTasks("rdagent", 10)) as any;
      const latestTask = (tasksResponse?.data || []).find((task: any) =>
        ["pending", "running", "cancelling", "completed", "failed"].includes(String(task.status || ""))
      );
      if (!latestTask?.task_id) {
        message.info("暂无可恢复的 RDAgent 挖掘任务");
        return;
      }
      await restoreRDAgentTask(latestTask.task_id, false);
    } catch (error: any) {
      console.error(error);
      message.error(error?.message || "加载最近 RDAgent 结果失败");
    } finally {
      setRdagentLoadingLatest(false);
    }
  };

  const clearRDAgentTaskResumeState = () => {
    clearPersistedRDAgentTaskStateAndBanner();
    if (!mining) {
      setAutoCampaignStatus((current) => (activeTab === "rdagent" ? current : current));
    }
    message.success("已清除 RDAgent 后台任务记录");
  };

  const cancelRDAgentTask = async () => {
    const taskId = autoCampaignStatus?.task_id || persistedRDAgentTaskState?.taskId;
    if (!taskId || taskId === "pending") {
      message.warning("当前没有可终止的 RDAgent 任务");
      return;
    }

    try {
      setLoading(true);
      await api.cancelRDAgentMining(taskId);
      clearTimers();
      setMining(false);
      setAutoCampaignStatus((current) =>
        current
          ? {
              ...current,
              status: "cancelled" as any,
              upstream_status: "rdagent_cancelled",
              error: "任务已终止",
              cancel_requested: true,
            }
          : current
      );
      clearPersistedRDAgentTaskStateAndBanner();
      message.success("已终止 RDAgent 任务");
    } catch (error: any) {
      message.error(error?.message || "终止 RDAgent 任务失败");
    } finally {
      setLoading(false);
    }
  };

  const startRDAgentContinueMining = async (continueRequest?: Record<string, any>) => {
    const payload = continueRequest?.payload;
    if (!payload?.objective) {
      message.error("当前 RDAgent 结果没有可继续挖掘的请求");
      return;
    }

    try {
      setRdagentContinuing(true);
      setLoading(true);
      setMining(true);
      setActiveTab("rdagent");
      setAutoCampaignResult(null);
      setAutoCampaignStatus({
        task_id: "pending",
        status: "pending",
        progress: 0,
        current_round: 0,
        total_rounds: Number(payload.max_iterations || 1),
        current_generation: 0,
        total_generations: getRDAgentTotalCandidates(payload),
        best_fitness: 0,
        avg_fitness: 0,
        retained_count: 0,
        fitness_history: { best: [], average: [] },
      });
      clearTimers();
      startClock();

      const response = (await api.startRDAgentMining(payload as any)) as any;
      const taskId = response?.data?.task_id;
      if (!taskId) throw new Error("未返回任务ID");
      syncPersistedRDAgentTaskState(taskId, "pending");
      await checkRDAgentProgress(taskId);
      pollRef.current = window.setInterval(() => checkRDAgentProgress(taskId), AUTO_PROGRESS_POLL_MS);
      message.success(response?.message || "RDAgent 继续挖掘任务已启动");
    } catch (error: any) {
      console.error(error);
      clearTimers();
      setMining(false);
      message.error(error?.message || "启动 RDAgent 继续挖掘失败");
    } finally {
      setLoading(false);
      setRdagentContinuing(false);
    }
  };

  const startContinueAutoMining = async (values: ContinueExplorationValues) => {
    if (!autoStatus?.task_id) {
      message.error("当前没有可继续探索的已完成任务");
      return;
    }

    try {
      const previousResult = autoResult;
      const previousStatus = autoStatus;
      const seedCurve = previousResult?.fitness_history || previousStatus?.fitness_history || { best: [], average: [] };
      const seedBest = (seedCurve.best || [0])[(seedCurve.best || [0]).length - 1] || 0;
      const seedAverage = (seedCurve.average || [0])[(seedCurve.average || [0]).length - 1] || 0;
      setAutoSeedState({
        result: previousResult || null,
        status: previousStatus || null,
      });
      setContinueExploring(true);
      setLoading(true);
      setMining(true);
      setAutoStatus({
        task_id: "pending",
        status: "pending",
        current_generation: seedCurve.best?.length ? 1 : 0,
        total_generations: Number(values.n_candidates || autoForm.getFieldValue("n_candidates") || 0),
        best_fitness: seedBest,
        avg_fitness: seedAverage,
        fitness_history: seedCurve,
      });
      clearTimers();
      startClock();
      const response = (await autoMiningApi.continueTask({
        parent_task_id: autoStatus.task_id,
        prompt: values.prompt,
        direction: values.direction,
        factor_update_mode: (values as any).factor_update_mode || "append",
        additional_base_factors: values.additional_base_factors || [],
        n_candidates: values.n_candidates,
        n_groups: values.n_groups,
        holding_period: values.holding_period,
        neutralize_industry: values.neutralize_industry,
        neutralize_cap: values.neutralize_cap,
      })) as any;
      const taskId = response?.data?.task_id;
      if (!taskId) throw new Error("未返回任务ID");
      await checkAutoProgress(taskId);
      pollRef.current = window.setInterval(() => checkAutoProgress(taskId), AUTO_PROGRESS_POLL_MS);
      message.success(response?.message || "继续探索任务已启动");
      setShowContinuePanel(false);
    } catch (error: any) {
      console.error(error);
      clearTimers();
      setMining(false);
      message.error(error?.message || "启动继续探索失败");
    } finally {
      setLoading(false);
      setContinueExploring(false);
    }
  };

  const handleContinueLLMSelectFactors = async () => {
    if (!autoStatus?.task_id) {
      message.error("当前没有可继续探索的已完成任务");
      return;
    }

    try {
      const values = await continueForm.validateFields(["prompt", "direction", "auto_additional_factor_count"]);
      setContinueSelectingFactors(true);
      setContinueSelectionSummary("");
      const response = (await autoMiningApi.selectContinueFactors({
        parent_task_id: autoStatus.task_id,
        prompt: values.prompt,
        direction: values.direction,
        factor_update_mode: continueForm.getFieldValue("factor_update_mode") || "append",
        max_factor_count: Number(values.auto_additional_factor_count || 5),
        candidate_limit: 80,
      })) as any;
      const selectedFactors = response?.data?.selected_factors || [];
      if (!selectedFactors.length) {
        throw new Error("LLM 未返回可用的基础因子");
      }
      continueForm.setFieldsValue({ additional_base_factors: selectedFactors });
      setContinueSelectionSummary(response?.data?.selection_rationale || "");
      message.success(
        response?.message ||
        ((continueForm.getFieldValue("factor_update_mode") || "append") === "reselect"
          ? `LLM 已根据上一轮报告重新选择 ${selectedFactors.length} 个基础因子`
          : `LLM 已根据上一轮报告自动补充 ${selectedFactors.length} 个新增因子`)
      );
    } catch (error: any) {
      message.error(error?.message || "自动选择基础因子失败");
    } finally {
      setContinueSelectingFactors(false);
    }
  };

  const checkAutoProgress = async (taskId: string) => {
    try {
      const response = (await autoMiningApi.getTaskStatus(taskId)) as any;
      if (!response.success) return;
      const status = response.data as MiningStatus;
      setAutoStatus(status);
      updateChart("progress", status.fitness_history, "研究曲线", "综合分数");
      if (status.status === "completed") {
        clearTimers();
        setMining(false);
        const result = (await autoMiningApi.getTaskResult(taskId)) as any;
        if (result.success) {
          setAutoResult({
            ...result.data,
            fitness_history: result.data?.fitness_history || status.fitness_history || { best: [], average: [] },
          });
          setAutoSeedState(null);
        }
      } else if (status.status === "failed") {
        clearTimers();
        setMining(false);
        setAutoSeedState(null);
        message.error(`自动挖掘失败: ${status.error || "未知错误"}`);
      }
    } catch (error) {
      console.error(error);
    }
  };

  const saveFactor = async (factor: ManualFactor | AutoFactor, index: number, retryCount = 0) => {
    const today = new Date();
    const dateStr = [
      today.getFullYear(),
      String(today.getMonth() + 1).padStart(2, "0"),
      String(today.getDate()).padStart(2, "0"),
      String(today.getHours()).padStart(2, "0"),
      String(today.getMinutes()).padStart(2, "0"),
      String(today.getSeconds()).padStart(2, "0"),
    ].join("");
    const stockCode = currentStockCode || "Unknown";
    const baseFactorName = `${activeTab === "manual" ? "Mined" : activeTab === "rdagent" ? "RDAgentMined" : "AutoMined"}_Factor_${index + 1}_${dateStr}_${stockCode}`;
    const factorName = retryCount === 0 ? baseFactorName : `${baseFactorName}_${retryCount}`;

    try {
      const rawExpr = factor.expression;
      const processedExpr = rawExpr
        .replace(/\bopen\b/g, "df['open']")
        .replace(/\bclose\b/g, "df['close']")
        .replace(/\bhigh\b/g, "df['high']")
        .replace(/\blow\b/g, "df['low']")
        .replace(/\bvolume\b/g, "df['volume']");

      const description = activeTab === "manual"
        ? `手动挖掘因子 | 表达式: ${rawExpr} | IC: ${(factor as ManualFactor).ic?.toFixed?.(4) || "-"} | IR: ${(factor as ManualFactor).ir?.toFixed?.(4) || "-"}`
        : `${activeTab === "rdagent" ? "RDAgent 挖掘因子" : "自动挖掘因子"} | 表达式: ${rawExpr} | Score: ${(factor as AutoFactor).score?.toFixed?.(1) || "-"} | Grade: ${(factor as AutoFactor).grade || "-"}`;

      const factorCode = `def calculate_factor(df):\n    \"\"\"${description}\"\"\"\n    import pandas as pd\n    import numpy as np\n    try:\n        result = ${processedExpr}\n        return result\n    except Exception:\n        return pd.Series(0, index=df.index)\n`;

      const normalizedAutoTaskDetails = isAutoLikeTab
        ? buildFactorTaskDetailsForDashboard(factor as AutoFactor)
        : null;
      const sourceTaskId = isAutoLikeTab
        ? ((factor as any).automation_meta?.round_task_id || autoStatus?.task_id || autoCampaignResult?.final_round_task_id)
        : undefined;

      const response = (await api.createFactor({
        name: factorName,
        code: factorCode,
        category: activeTab === "manual" ? "遗传挖掘" : activeTab === "rdagent" ? "RDAgent 挖掘" : "自动挖掘",
        description,
        formula_type: "function",
        scope_type: activeTab === "manual" ? "stock" : "universe",
        target_stock_code: activeTab === "manual" ? String(manualForm.getFieldValue("stock_code") || "").trim() : "",
        target_universe: isAutoLikeTab ? String((activeTab === "rdagent" ? rdAgentForm : autoForm).getFieldValue("universe") || "").trim() : "",
        origin_type: activeTab === "manual" ? "genetic_mining" : activeTab === "rdagent" ? "rdagent_mining" : "auto_mining",
        task_metadata: isAutoLikeTab ? {
          task_id: sourceTaskId,
          source: (factor as AutoFactor).source || (activeTab === "rdagent" ? "factorhub_rdagent_campaign" : "factorhub_auto_iteration"),
          task_details: {
            ...(normalizedAutoTaskDetails || {}),
            expression: normalizedAutoTaskDetails?.expression || rawExpr,
          },
        } : {
          source_expression: rawExpr,
          params: {
            stock_code: String(manualForm.getFieldValue("stock_code") || "").trim(),
            base_factors: manualForm.getFieldValue("base_factors") || [],
          },
        },
      })) as any;

      if (response.success) {
        message.success(`因子 "${factorName}" 已保存到因子库`);
        setSavedFactorNames((prev) => new Set(prev).add(factorName));
        await loadFactors();
        return;
      }
      throw new Error(response?.data?.detail || response?.message || "未知错误");
    } catch (error: any) {
      const errorMsg = error?.response?.data?.detail || error?.message || "未知错误";
      if (String(errorMsg).includes("已存在") && retryCount < 5) {
        await saveFactor(factor, index, retryCount + 1);
      } else {
        message.error(`保存因子失败: ${errorMsg}`);
      }
    }
  };

  const saveAllFactors = async () => {
    const factorsToSave = (
      activeTab === "manual"
        ? manualResult?.factors
        : activeTab === "rdagent"
          ? autoCampaignResult?.retained_factors
          : autoResult?.factors
    ) || [];
    if (!factorsToSave.length) {
      message.warning("没有可保存的因子");
      return;
    }
    for (let i = 0; i < factorsToSave.length; i += 1) {
      // eslint-disable-next-line no-await-in-loop
      await saveFactor(factorsToSave[i] as any, i);
    }
  };

  const selectedFactorCount = useMemo(() => {
    const values = activeTab === "manual" ? watchedManualBaseFactors : watchedAutoBaseFactors;
    return values?.length || 0;
  }, [activeTab, watchedAutoBaseFactors, watchedManualBaseFactors]);

  const updateAutoBaseFactors = (baseFactors: string[]) => {
    const nextValues = { ...autoForm.getFieldsValue(), base_factors: baseFactors };
    autoForm.setFieldsValue({ base_factors: baseFactors });
    persistAutoMiningForm({}, nextValues);
  };

  const updateAutoLaunchMode = (mode: "single" | "campaign") => {
    setAutoLaunchMode(mode);
    persistAutoMiningForm({}, { ...autoForm.getFieldsValue(), preferred_launch_mode: mode });
  };

  const persistAutoMiningForm = (_changedValues: any, allValues: any) => {
    if (typeof window === "undefined") return;
    try {
      const payload = {
        ...allValues,
        preferred_launch_mode: allValues?.preferred_launch_mode || autoLaunchMode,
        dateRange:
          Array.isArray(allValues?.dateRange) && allValues.dateRange.length === 2
            ? [allValues.dateRange[0]?.format?.("YYYY-MM-DD"), allValues.dateRange[1]?.format?.("YYYY-MM-DD")]
            : null,
      };
      window.localStorage.setItem(AUTO_MINING_FORM_STORAGE_KEY, JSON.stringify(payload));
    } catch (error) {
      console.warn("保存自动挖掘本地配置失败:", error);
    }
  };

  const isOpenableReportUrl = (reportUrl?: string) => {
    if (!reportUrl) return false;
    return /^https?:\/\//i.test(reportUrl) || reportUrl.startsWith("/api/");
  };

  const resolveReportUrl = (reportUrl?: string) => {
    const url = reportUrl || "";
    if (!isOpenableReportUrl(url)) return "";
    return resolveApiUrl(url);
  };

  const getRDAgentStageLabel = (upstreamStatus?: string) => {
    const status = String(upstreamStatus || "");
    if (status.includes("cli_starting")) return "内部 RDAgent Loop 启动";
    if (status.includes("cli_completed")) return "内部 RDAgent Loop 完成";
    if (status.includes("cli_failed")) return "内部 RDAgent Loop 失败";
    if (status.includes("hypothesis")) return "Hypothesis 生成研究假设";
    if (status.includes("experiment")) return "Experiment 生成 factor tasks";
    if (status.includes("coding")) return "Coding 承接表达式实现";
    if (status.includes("running")) return "Running 回测与评分";
    if (status.includes("feedback")) return "Feedback 反馈优化";
    if (status.includes("completed")) return "Trace 完成并沉淀结果";
    if (status.includes("failed")) return "Failed 执行失败";
    return "Pending 等待启动";
  };

  const getRDAgentStageItems = (activeStage?: string) => {
    const stageKeys = ["hypothesis", "experiment", "coding", "running", "feedback", "trace"];
    const normalizedStage = String(activeStage || "").includes("cli_completed")
      ? "trace"
      : String(activeStage || "").includes("cli")
        ? "running"
        : activeStage;
    const activeIndex = Math.max(
      0,
      stageKeys.findIndex((key) => String(normalizedStage || "").includes(key)),
    );
    return [
      { key: "hypothesis", title: "Hypothesis", description: "从目标和反馈生成下一轮研究假设" },
      { key: "experiment", title: "Experiment", description: "LLM 根据 hypothesis 生成 factor tasks" },
      { key: "coding", title: "Coding", description: "映射为 FactorHub 可计算表达式" },
      { key: "running", title: "Running", description: "复用回测/评分体系计算指标" },
      { key: "feedback", title: "Feedback", description: "根据阈值接收或拒绝候选" },
      { key: "trace", title: "Trace", description: "沉淀 SOTA 候选与下一轮线索" },
    ].map((stage, index) => ({
      ...stage,
      state: index < activeIndex ? "done" : index === activeIndex ? "active" : "pending",
    }));
  };

  const getRDAgentCandidateDiagnostics = (factor: any) => {
    const rdagentDetails = factor?.task_details?.rdagent || {};
    const score = rdagentDetails.candidate_score || {};
    const failureReasons = rdagentDetails.policy_failure_reasons || factor?.policy_diagnostics?.failure_reasons || [];
    const diagnostics = [];
    if (rdagentDetails.adapter_normalized) {
      diagnostics.push({
        type: "info",
        label: "表达式已规范化",
        text: rdagentDetails.raw_expression
          ? `已从 RDAgent 原始表达式规范化为 FactorHub 语法：${rdagentDetails.raw_expression}`
          : "已通过 adapter 规范化为 FactorHub 可解析语法",
      });
    }
    if (rdagentDetails.parser_repair_error) {
      diagnostics.push({
        type: "warning",
        label: "Parser 前修复",
        text: `首次解析失败后已按 FactorHub 表达式契约修复：${rdagentDetails.parser_repair_error}`,
      });
    }
    if (rdagentDetails.evaluation_error) {
      diagnostics.push({
        type: "error",
        label: "回测失败",
        text: String(rdagentDetails.evaluation_error),
      });
    }
    if (score.report_url || factor?.report_url) {
      diagnostics.push({
        type: "success",
        label: "报告",
        text: "已生成 FactorHub 回测报告",
      });
    }
    if (failureReasons.length) {
      diagnostics.push({
        type: "warning",
        label: "阈值诊断",
        text: failureReasons
          .map((item: any) => `${item.label || item.rule}: ${Number(item.value ?? 0).toFixed(3)} / ${Number(item.threshold ?? 0).toFixed(3)}`)
          .join("；"),
      });
    }
    if (!diagnostics.length && factor?.status && factor.status !== "computed") {
      diagnostics.push({
        type: factor.status === "accepted" ? "success" : "info",
        label: "状态",
        text: factor.status === "accepted" ? "已通过入选策略" : "已完成 FactorHub 回测评估",
      });
    }
    return diagnostics;
  };

  const toggleReportPreview = (reportUrl: string) => {
    const resolvedUrl = resolveReportUrl(reportUrl);
    if (!resolvedUrl) {
      message.info("该报告引用不是可预览的 HTML URL");
      return;
    }
    const nextUrl = expandedReportUrl === resolvedUrl ? null : resolvedUrl;
    setExpandedReportUrl(nextUrl);
    if (nextUrl && typeof window !== "undefined") {
      window.setTimeout(() => {
        const el = document.getElementById(`report-preview-${encodeURIComponent(resolvedUrl)}`);
        el?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 120);
    }
  };

  const toggleDetailsPreview = (factor: AutoFactor, index: number) => {
    const detailKey = getFactorDetailKey(factor, index);
    const nextKey = expandedDetailsKey === detailKey ? null : detailKey;
    setExpandedDetailsKey(nextKey);
    if (nextKey && typeof window !== "undefined") {
      window.setTimeout(() => {
        const el = document.getElementById(`factor-details-${encodeURIComponent(detailKey)}`);
        el?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 120);
    }
  };

  const isRDAgentExpressionFormatFailure = (error?: string) => {
    const text = String(error || "");
    return activeTab === "rdagent" && (
      text.includes("FactorHub 表达式契约")
      || text.includes("无法解析的表达式")
      || text.includes("RDAgent formulation")
      || text.includes("不支持的字段或变量")
      || text.includes("不支持的函数")
    );
  };

  const renderRDAgentLoopOverview = (upstreamStatus?: string) => (
    <div className="rdagent-loop-overview">
      {getRDAgentStageItems(upstreamStatus).map((stage) => (
        <div key={stage.key} className={`rdagent-stage rdagent-stage-${stage.state}`}>
          <div className="rdagent-stage-dot" />
          <div className="rdagent-stage-title">{stage.title}</div>
          <div className="rdagent-stage-desc">{stage.description}</div>
        </div>
      ))}
    </div>
  );

  const renderRDAgentTrace = (rounds: RDAgentTraceRound[] = []) => {
    if (!rounds.length) {
      return <Alert message="暂无 RDAgent trace；任务完成后会展示每轮 hypothesis、experiment、evaluation 和 feedback。" type="info" showIcon />;
    }

    return (
      <div className="rdagent-trace-list">
        {rounds.map((round) => {
          const hypothesis = round.hypothesis || {};
          const feedback = round.feedback || {};
          const evaluation = round.evaluation || {};
          const candidates = round.candidates || round.all_factors || [];
          return (
            <Card key={`${round.task_id}-${round.round_index}`} className="rdagent-trace-card" size="small">
              <div className="rdagent-trace-header">
                <Space wrap>
                  <Tag color="blue">Round {round.round_index}</Tag>
                  <Tag color={feedback.hypothesis_evaluation === "supported" ? "green" : "orange"}>
                    {feedback.hypothesis_evaluation === "supported" ? "Supported" : "Needs Next Hypothesis"}
                  </Tag>
                  <Tag color="purple">{hypothesis.research_direction || "simple_baseline"}</Tag>
                </Space>
                <div className="rdagent-trace-score">Best {Number(round.best_score || 0).toFixed(1)}</div>
              </div>

              <div className="rdagent-trace-section">
                <div className="rdagent-trace-label">Hypothesis</div>
                <div className="rdagent-trace-text">{hypothesis.statement || "暂无研究假设"}</div>
              </div>

              <Row gutter={[12, 12]}>
                <Col xs={24} md={12}>
                  <div className="rdagent-trace-section">
                    <div className="rdagent-trace-label">Experiment Inputs</div>
                    <Space wrap>
                      {(round.input_base_factors || []).map((name) => <Tag key={`${round.round_index}-${name}`}>{name}</Tag>)}
                      {!(round.input_base_factors || []).length ? <span className="text-hint">无基础因子上下文</span> : null}
                    </Space>
                  </div>
                </Col>
                <Col xs={24} md={12}>
                  <div className="rdagent-trace-section">
                    <div className="rdagent-trace-label">Evaluation</div>
                    <Space wrap>
                      <Tag color="green">Avg {Number(round.avg_score || 0).toFixed(1)}</Tag>
                      <Tag>候选 {candidates.length}</Tag>
                      <Tag color="cyan">保留 {round.retained_count || 0}</Tag>
                    </Space>
                    {evaluation.report_ref ? <div className="rdagent-trace-muted">{evaluation.report_ref}</div> : null}
                  </div>
                </Col>
              </Row>

              <div className="rdagent-candidate-grid">
                {candidates.map((factor: any, index: number) => {
                  const score = factor.task_details?.rdagent?.candidate_score || {};
                  const diagnostics = getRDAgentCandidateDiagnostics(factor);
                  return (
                    <div key={`${factor.candidate_id || factor.name}-${index}`} className={`rdagent-candidate rdagent-candidate-${factor.status || "computed"}`}>
                      <div className="rdagent-candidate-top">
                        <span>{factor.name || `Candidate ${index + 1}`}</span>
                        <Tag color={factor.status === "accepted" ? "green" : factor.status === "rejected" ? "red" : "default"}>
                          {factor.status || "computed"}
                        </Tag>
                      </div>
                      <div className="factor-expression">{factor.expression}</div>
                      <Space wrap size={4}>
                        <Tag>rankIC {Number(score.rank_ic ?? factor.report_metrics?.rank_ic ?? 0).toFixed(3)}</Tag>
                        <Tag>Sharpe {Number(score.sharpe ?? factor.report_metrics?.sharpe ?? 0).toFixed(2)}</Tag>
                        <Tag>Coverage {Number(score.valid_coverage ?? 0).toFixed(2)}</Tag>
                        <Tag>Corr {Number(score.max_correlation_with_sota ?? 0).toFixed(2)}</Tag>
                      </Space>
                      {diagnostics.length ? (
                        <div className="rdagent-candidate-diagnostics">
                          {diagnostics.map((item, diagIndex) => (
                            <Alert
                              key={`${factor.candidate_id || factor.name}-diagnostic-${diagIndex}`}
                              type={item.type as any}
                              showIcon
                              message={item.label}
                              description={item.text}
                            />
                          ))}
                        </div>
                      ) : null}
                    </div>
                  );
                })}
              </div>

              <Alert
                type={feedback.hypothesis_evaluation === "supported" ? "success" : "warning"}
                showIcon
                message="Feedback"
                description={
                  <Space direction="vertical" size={4}>
                    <span>{feedback.observations || "暂无反馈"}</span>
                    {feedback.next_hypothesis ? <span>下一轮：{feedback.next_hypothesis}</span> : null}
                  </Space>
                }
              />
            </Card>
          );
        })}
      </div>
    );
  };

  const renderRDAgentManualReport = (report?: Record<string, any> | null) => {
    if (!report) {
      return <Alert message="暂无手动报告格式结果；请重新运行一次 RDAgent 挖掘。" type="info" showIcon />;
    }

    const sections = Array.isArray(report.sections) ? report.sections : [];
    const ranking = sections.find((section: any) => section.id === "candidate-ranking");
    const evidence = sections.find((section: any) => section.id === "backtest-evidence");
    const nextPlan = sections.find((section: any) => section.id === "next-mining-plan");
    const updates = sections.find((section: any) => section.id === "factorhub-updates");
    const reportUrl = report.html_report_url || report.location;
    const resolvedReportUrl = resolveReportUrl(String(reportUrl || ""));

    return (
      <Card size="small" title={report.title || "RDAgent 手动挖掘报告"} style={{ marginBottom: 16, background: "#fafafa" }}>
        <Space direction="vertical" size={12} style={{ width: "100%" }}>
          <Space wrap>
            <Tag color="blue">{report.layout || "manual-mining-report"}</Tag>
            {reportUrl ? <Tag color={resolvedReportUrl ? "cyan" : "default"}>{resolvedReportUrl || `内部引用：${reportUrl}`}</Tag> : null}
            {report.iteration ? <Tag>第 {report.iteration} 轮</Tag> : null}
            {resolvedReportUrl ? (
              <Button size="small" onClick={() => toggleReportPreview(String(reportUrl))}>
                {expandedReportUrl === resolvedReportUrl ? "收起报告" : "打开报告"}
              </Button>
            ) : null}
          </Space>

          {resolvedReportUrl && expandedReportUrl === resolvedReportUrl ? (
            <div id={`report-preview-${encodeURIComponent(resolvedReportUrl)}`} style={{ border: "1px solid #e5e7eb", borderRadius: 8, overflow: "hidden", background: "#fff" }}>
              <iframe
                title={`${report.title || "RDAgent 手动挖掘报告"} 预览`}
                src={resolvedReportUrl}
                style={{ width: "100%", height: 520, border: 0, display: "block" }}
              />
            </div>
          ) : null}

          {ranking ? (
            <div>
              <div className="rdagent-trace-label">{ranking.title || "候选因子排序"}</div>
              <div className="rdagent-trace-muted" style={{ marginBottom: 8 }}>{ranking.summary}</div>
              <div className="rdagent-candidate-grid">
                {(ranking.rows || []).map((row: any, index: number) => {
                  const metrics = row.metrics || {};
                  return (
                    <div key={`${row.candidate_id || row.name}-${index}`} className={`rdagent-candidate rdagent-candidate-${row.status || "computed"}`}>
                      <div className="rdagent-candidate-top">
                        <span>{row.name || `Candidate ${index + 1}`}</span>
                        <Tag color={row.status === "accepted" ? "green" : row.status === "rejected" ? "red" : "default"}>{row.status || "computed"}</Tag>
                      </div>
                      <div className="factor-expression">{row.expression}</div>
                      <Space wrap size={4}>
                        {row.optimization_metric ? <Tag color="blue">{row.optimization_metric} {Number(row.optimization_score ?? 0).toFixed(3)}</Tag> : null}
                        <Tag>rankIC {Number(metrics.rank_ic ?? 0).toFixed(3)}</Tag>
                        <Tag>Sharpe {Number(metrics.sharpe ?? 0).toFixed(2)}</Tag>
                        <Tag>Coverage {Number(metrics.valid_coverage ?? 0).toFixed(2)}</Tag>
                        <Tag>Corr {Number(metrics.max_correlation_with_sota ?? 0).toFixed(2)}</Tag>
                      </Space>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : null}

          {evidence ? (
            <Alert
              type="info"
              showIcon
              message={evidence.title || "回测证据"}
              description={
                <Space direction="vertical" size={4}>
                  <span>{evidence.summary}</span>
                  <Space wrap>
                    {evidence.metrics?.rank_ic !== undefined ? <Tag color="green">rankIC {Number(evidence.metrics.rank_ic).toFixed(3)}</Tag> : null}
                    {evidence.metrics?.annualized_return !== undefined ? <Tag>Annualized {Number(evidence.metrics.annualized_return).toFixed(3)}</Tag> : null}
                    {evidence.metrics?.max_drawdown !== undefined ? <Tag color="orange">Drawdown {Number(evidence.metrics.max_drawdown).toFixed(3)}</Tag> : null}
                  </Space>
                </Space>
              }
            />
          ) : null}

          {updates ? (
            <Alert
              type={(updates.updates || []).length ? "success" : "warning"}
              showIcon
              message={updates.title || "FactorHub 分类建议"}
              description={updates.summary}
            />
          ) : null}

          {nextPlan ? (
            <Alert
              type="success"
              showIcon
              message={nextPlan.title || "继续挖掘建议"}
              description={
                <Space direction="vertical" size={4}>
                  <span>{nextPlan.summary || "暂无下一轮建议"}</span>
                  {nextPlan.next_action ? <Tag color="purple">{nextPlan.next_action}</Tag> : null}
                </Space>
              }
            />
          ) : null}
        </Space>
      </Card>
    );
  };

  const renderRDAgentContinueAction = (continueRequest?: Record<string, any> | null) => {
    const payload = continueRequest?.payload || {};
    if (!continueRequest) {
      return null;
    }

    return (
      <Card size="small" title="继续挖掘" style={{ marginBottom: 16 }}>
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 12 }}
          message="复用本轮报告和 feedback 开启下一轮"
          description={
            <Space direction="vertical" size={4}>
              <span>{payload.objective || "根据上一轮结果继续优化 RDAgent 假设。"}</span>
            </Space>
          }
        />
        <Button
          type="primary"
          icon={<PlayCircleOutlined />}
          loading={rdagentContinuing}
          disabled={mining}
          onClick={() => void startRDAgentContinueMining(continueRequest)}
        >
          继续 RDAgent 挖掘
        </Button>
      </Card>
    );
  };

  const renderFactorOptions = () =>
    factors.map((factor) => (
      <Option key={factor.id} value={factor.name} label={factor.name}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontWeight: 500 }}>{factor.name}</span>
            <Tag color={factor.source === "preset" ? "success" : "warning"}>{factor.source === "preset" ? "预置" : "自定义"}</Tag>
            <Tag color="blue">{factor.category}</Tag>
          </div>
          <div style={{ fontSize: 12, color: "#64748b", fontFamily: "monospace" }}>{factor.code}</div>
          {factor.description && <div style={{ fontSize: 12, color: "#94a3b8" }}>{factor.description}</div>}
        </div>
      </Option>
    ));

  const renderAutoModeSelector = () => (
    <Card size="small" className="auto-mode-card" style={{ marginBottom: 16 }}>
      <Space direction="vertical" size={12} style={{ width: "100%" }}>
        <div>
          <div className="auto-mode-title">研究模式</div>
          <div className="auto-mode-copy">
            自动挖掘统一使用同一个入口：先配置基础因子和研究目标，再选择是做单轮验证，还是让系统连续探索并合并展示 QuantGPT 与 FactorHub 报告。
          </div>
        </div>
        <Segmented
          block
          value={autoLaunchMode}
          onChange={(value) => updateAutoLaunchMode(value as "single" | "campaign")}
          options={[
            {
              label: (
                <div className="auto-mode-option">
                  <div className="auto-mode-option-title">单轮研究</div>
                  <div className="auto-mode-option-copy">快速生成一轮候选并验证报告</div>
                </div>
              ),
              value: "single",
            },
            {
              label: (
                <div className="auto-mode-option">
                  <div className="auto-mode-option-title">连续探索</div>
                  <div className="auto-mode-option-copy">多轮迭代筛选并保留最终候选</div>
                </div>
              ),
              value: "campaign",
            },
          ]}
        />
      </Space>
    </Card>
  );

  const renderRDAgentMiningTab = () => (
    <Row gutter={[24, 24]}>
      <Col xs={24} lg={8}>
        <Card title="RDAgent 因子挖掘配置" className="config-card">
          {persistedRDAgentTaskState?.taskId && isRDAgentResumableStatus(persistedRDAgentTaskState.status) ? (
            <Alert
              type="success"
              showIcon
              style={{ marginBottom: 16 }}
              message="检测到后台 RDAgent 任务"
              description={`页面重新进入时会自动恢复任务状态。当前记录任务 ID：${persistedRDAgentTaskState.taskId}`}
              action={
                <Button size="small" onClick={clearRDAgentTaskResumeState}>
                  清除记录
                </Button>
              }
            />
          ) : null}
          <Button
            block
            style={{ marginBottom: 16 }}
            loading={rdagentLoadingLatest}
            disabled={mining}
            onClick={() => void loadLatestRDAgentResult()}
          >
            加载最近 RDAgent 结果
          </Button>
          <Button
            block
            danger
            icon={<ExclamationCircleOutlined />}
            style={{ marginBottom: 16 }}
            disabled={!autoCampaignStatus?.task_id || ["completed", "failed", "cancelled"].includes(String(autoCampaignStatus?.status || ""))}
            loading={loading && mining}
            onClick={() => void cancelRDAgentTask()}
          >
            终止当前 RDAgent 任务
          </Button>
          <Alert
            type="info"
            showIcon
            style={{ marginBottom: 16 }}
            message="FactorHub 表达式格式"
            description="RDAgent 会生成单行表达式：字段限定 open/high/low/close/volume/amount/vwap/pct_change；rank/tanh/log/abs/sigmoid 只接 1 个参数；ts_mean/ts_std/ts_zscore/decay_linear/ts_min/ts_max 使用 (表达式, 整数窗口)；比较两个序列请用 min(x, y) / max(x, y)。常见 Ref/Mean/Std/If/turnover 写法会先自动规范化，仍无法解析时会直接失败并显示格式原因。"
          />
          <Form form={rdAgentForm} layout="vertical" onFinish={startRDAgentMining}>
            <Divider styles={{ content: { margin: 0 } }} titlePlacement="left">任务目标</Divider>
            <Form.Item label="挖掘目标" name="objective" rules={[{ required: true, message: "请输入挖掘目标" }]}>
              <Input.TextArea rows={3} placeholder="例如：寻找低相关、低换手、rankIC 稳定的价量因子" />
            </Form.Item>
            <Form.Item label="日期范围" name="dateRange" rules={[{ required: true, message: "请选择日期范围" }]}>
              <RangePicker
                style={{ width: "100%" }}
                allowClear
                inputReadOnly={isMobileView}
                placement={isMobileView ? "bottomLeft" : undefined}
                getPopupContainer={(trigger) => trigger.parentElement || document.body}
              />
            </Form.Item>
            <Row gutter={16}>
              <Col xs={24} sm={12}>
                <Form.Item label="股票池" name="universe" rules={[{ required: true, message: "请选择股票池" }]}>
                  <Select>
                    <Option value="hs300">HS300</Option>
                    <Option value="zz500">ZZ500</Option>
                    <Option value="zz1000">ZZ1000</Option>
                    <Option value="all">全市场</Option>
                  </Select>
                </Form.Item>
              </Col>
              <Col xs={24} sm={12}>
                <Form.Item label="基准" name="benchmark" rules={[{ required: true, message: "请选择基准" }]}>
                  <Select>
                    <Option value="hs300">HS300</Option>
                    <Option value="zz500">ZZ500</Option>
                    <Option value="zz1000">ZZ1000</Option>
                  </Select>
                </Form.Item>
              </Col>
            </Row>

            <Divider styles={{ content: { margin: 0 } }} titlePlacement="left">启动配置</Divider>
            <div className="rdagent-bootstrap-mode">
              <div className="rdagent-bootstrap-mode-head">
                <div>
                  <div className="rdagent-bootstrap-mode-title">字段与基础因子来源</div>
                  <div className="rdagent-bootstrap-mode-desc">
                    候选字段决定 `LLM` 可用输入空间；基础因子会作为研究上下文输入，不是每次都从零开始。
                  </div>
                </div>
                <Segmented
                  value={rdagentBootstrapMode}
                  onChange={(value) => setRdagentBootstrapMode(value as RDAgentBootstrapMode)}
                  options={[
                    { label: "LLM 自动配置", value: "llm_auto" },
                    { label: "手动配置", value: "manual" },
                  ]}
                />
              </div>
              {rdagentBootstrapMode === "llm_auto" ? (
                <div className="rdagent-bootstrap-panel">
                  <Form.Item style={{ marginBottom: 12 }}>
                    <Button block type="primary" onClick={() => void handleRDAgentLLMBootstrap()} loading={llmSelectingFactors}>
                      {rdagentSelectionSummary ? "重新生成候选字段与基础因子" : "生成候选字段与基础因子"}
                    </Button>
                  </Form.Item>
                  <div className="text-hint" style={{ marginBottom: 12 }}>
                    先根据目标自动挑选最小充分字段集合，再从因子库导入一组更合适的基础因子作为起点。
                  </div>
                  {renderRDAgentBootstrapSummary()}
                </div>
              ) : (
                <Alert
                  type="info"
                  showIcon
                  style={{ marginBottom: 16 }}
                  message="当前为手动配置"
                  description="你自己决定字段 Universe 和基础因子；RDAgent 会在这组边界内继续生成与优化。"
                />
              )}
            </div>

            <Divider styles={{ content: { margin: 0 } }} titlePlacement="left">RDAgent Loop</Divider>
            <Row gutter={16}>
              <Col xs={24} sm={12}>
                <Form.Item label="最大优化轮数" name="max_iterations" rules={[{ required: true, message: "请输入最大优化轮数" }]}>
                  <InputNumber min={1} max={6} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12}>
                <Form.Item label="每轮候选" name="candidates_per_iteration" rules={[{ required: true, message: "请输入每轮候选数" }]}>
                  <InputNumber min={1} max={5} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
            </Row>
            <Row gutter={16}>
              <Col xs={24} sm={12}>
                <Form.Item label="分组数" name="n_groups">
                  <InputNumber min={2} max={20} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12}>
                <Form.Item label="持有期" name="holding_period">
                  <InputNumber min={1} max={60} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
            </Row>
            <Form.Item label="优化方向" name="direction">
              <Select allowClear options={OPTIMIZATION_DIRECTION_OPTIONS} />
            </Form.Item>
            <Row gutter={16}>
              <Col xs={24} sm={12}>
                <Form.Item name="neutralize_industry" valuePropName="checked">
                  <Checkbox>行业中性</Checkbox>
                </Form.Item>
              </Col>
              <Col xs={24} sm={12}>
                <Form.Item name="neutralize_cap" valuePropName="checked">
                  <Checkbox>市值中性</Checkbox>
                </Form.Item>
              </Col>
            </Row>
            <Form.Item
              label="候选字段 Universe"
              name="candidate_universe"
              tooltip="RDAgent 只会在这里列出的字段范围内组合表达式；自动模式下可先让 LLM 生成，再人工微调。"
              rules={[{ required: true, message: "请选择候选字段" }]}
            >
              <Select mode="multiple" placeholder="选择 FactorHub 已有行情字段" maxTagCount="responsive">
                {FACTORHUB_MARKET_FIELDS.map((field) => (
                  <Option key={field} value={field}>{field}</Option>
                ))}
              </Select>
            </Form.Item>
            <Form.Item
              label="基础因子"
              name="base_factors"
              tooltip="这些基础因子会作为初始研究上下文输入给 RDAgent，用来继承已有研究，而不是每次全新构建。"
            >
              <Select mode="multiple" placeholder="可选：作为 RDAgent 初始研究上下文" showSearch optionLabelProp="label" maxTagCount="responsive">
                {renderFactorOptions()}
              </Select>
            </Form.Item>
            {rdagentBootstrapMode === "llm_auto" ? (
              <div className="text-hint" style={{ marginTop: -8, marginBottom: 12 }}>
                自动模式生成后，你仍然可以在这里继续删改字段和基础因子。
              </div>
            ) : null}
            <Form.Item label="SOTA 分类 ID" name="sota_library_id">
              <Input placeholder="factorhub-sota" />
            </Form.Item>

            <Divider styles={{ content: { margin: 0 } }} titlePlacement="left">入选阈值</Divider>
            <Row gutter={16}>
              <Col xs={24} sm={12}>
                <Form.Item label="最低 rankIC" name="min_rank_ic">
                  <InputNumber min={-1} max={1} step={0.001} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12}>
                <Form.Item label="最低年化收益增量" name="min_annualized_return_delta">
                  <InputNumber min={-1} max={1} step={0.001} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
            </Row>
            <Row gutter={16}>
              <Col xs={24} sm={12}>
                <Form.Item label="最大回撤退化" name="max_drawdown_regression">
                  <InputNumber min={0} max={1} step={0.01} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12}>
                <Form.Item label="最低有效覆盖率" name="min_valid_coverage">
                  <InputNumber min={0} max={1} step={0.01} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
            </Row>
            <Form.Item label="与 SOTA 最大相关性" name="max_correlation_with_sota">
              <InputNumber min={0} max={1} step={0.01} style={{ width: "100%" }} />
            </Form.Item>
            <Form.Item>
              <Button type="primary" htmlType="submit" icon={<PlayCircleOutlined />} loading={loading && activeTab === "rdagent"} block size="large" disabled={mining}>
                {mining && activeTab === "rdagent" ? "挖掘中..." : "启动 RDAgent 因子挖掘"}
              </Button>
            </Form.Item>
          </Form>
        </Card>
      </Col>

      <Col xs={24} lg={16}>
        <Card title="RDAgent 因子挖掘结果" className="result-card">{renderStatusCard()}</Card>
      </Col>
    </Row>
  );

  const renderStatusCard = () => {
    const renderReadyState = (title: string, description: string) => (
      <div style={{ padding: 48, textAlign: "center" }}>
        <RocketOutlined style={{ fontSize: 48, color: "#94a3b8", marginBottom: 16 }} />
        <h3 style={{ marginBottom: 8 }}>{title}</h3>
        <p style={{ color: "#64748b", margin: 0 }}>{description}</p>
      </div>
    );

    if (!mining && !currentStatus && !currentResult) {
      if (isAutoLikeTab && autoWorkflowMode === "campaign" && autoCampaignResult) {
        // fall through to dedicated campaign result rendering below
      } else if (isAutoLikeTab && autoWorkflowMode === "campaign" && autoCampaignStatus) {
        // fall through to dedicated campaign running rendering below
      } else {
        return renderReadyState(
          `准备开始${activeTab === "manual" ? "手动挖掘" : activeTab === "rdagent" ? " RDAgent 挖掘" : "自动挖掘"}`,
          activeTab === "manual"
            ? "选择基础因子后即可启动遗传算法。"
            : activeTab === "rdagent"
              ? "设置目标、候选字段和筛选条件后即可启动。"
              : "导入基础因子并设置参数后即可启动。"
        );
      }
    }

    if (isAutoLikeTab && autoWorkflowMode === "campaign") {
      if (!mining && (!autoCampaignStatus || autoCampaignStatus.status === "pending")) {
        return renderReadyState(
          activeTab === "rdagent" ? "等待启动 RDAgent 挖掘" : "等待启动自动化挖掘",
          activeTab === "rdagent"
            ? "左侧完成目标与筛选配置后即可启动。"
            : "左侧完成自动化参数配置后即可启动。"
        );
      }

      if (!mining && autoCampaignStatus?.status === "failed") {
        const isRDAgentDataBacktestFailure = activeTab === "rdagent"
          && String(autoCampaignStatus.error || "").includes("FactorHub 内部数据集和回测引擎");
        const isRDAgentExpressionFailure = isRDAgentExpressionFormatFailure(autoCampaignStatus.error);
        return (
          <Alert
            type="error"
            showIcon
            message={
              isRDAgentDataBacktestFailure
                ? "FactorHub 数据/回测链路未通过"
                : isRDAgentExpressionFailure
                  ? "RDAgent 表达式格式未通过"
                : activeTab === "rdagent" ? "RDAgent 挖掘失败" : "自动化挖掘失败"
            }
            description={
              <Space direction="vertical" size={4}>
                <span>{autoCampaignStatus.error || "任务失败，但后端没有返回详细错误。"}</span>
                {isRDAgentDataBacktestFailure ? (
                  <span>候选因子没有产生可验证的内部行情数据、回测指标或报告，因此页面不会展示伪完成结果。</span>
                ) : null}
                {isRDAgentExpressionFailure ? (
                  <span>请让 RDAgent 输出单行 FactorHub 表达式：只使用行情字段和支持函数，一元函数只传 1 个参数，窗口函数使用整数窗口，两个序列比较用 min(x, y) / max(x, y)。</span>
                ) : null}
                <span>任务 ID：{autoCampaignStatus.task_id}</span>
              </Space>
            }
          />
        );
      }

      if (!mining && autoCampaignStatus?.status === "cancelled") {
        return renderReadyState(
          activeTab === "rdagent" ? "RDAgent 挖掘已停止" : "自动化挖掘已停止",
          "可以调整左侧配置后重新启动。"
        );
      }

      if (!mining && autoCampaignStatus?.status === "completed" && !autoCampaignResult) {
        return (
          <Alert
            type="info"
            showIcon
            message={activeTab === "rdagent" ? "RDAgent 结果等待加载" : "自动化结果等待加载"}
            description={
              <Space direction="vertical" size={4}>
                <span>任务已完成，但结果还没有恢复到当前页面。</span>
                <span>任务 ID：{autoCampaignStatus.task_id}</span>
              </Space>
            }
          />
        );
      }

      if (!mining && autoCampaignStatus?.status === "running" && autoCampaignStatus.progress >= 100 && !autoCampaignResult) {
        return (
          <Alert
            type="warning"
            showIcon
            message={activeTab === "rdagent" ? "RDAgent 结果未完成写入" : "挖掘结果未完成写入"}
            description={
              <Space direction="vertical" size={4}>
                <span>任务已经到达 100%，但后端还没有返回可展示结果。请稍后刷新，或重新启动任务。</span>
                <span>任务 ID：{autoCampaignStatus.task_id}</span>
              </Space>
            }
          />
        );
      }

      if (mining && autoCampaignStatus) {
        const progressPercent = autoCampaignStatus.progress || 0;
        const runningRDAgentCandidates = activeTab === "rdagent" ? (autoCampaignStatus.candidates || []) : [];
        const runningRDAgentRound = activeTab === "rdagent" && autoCampaignStatus.latest_round
          ? {
              ...autoCampaignStatus.latest_round,
              candidates: runningRDAgentCandidates.length
                ? runningRDAgentCandidates
                : (autoCampaignStatus.latest_round.candidates || autoCampaignStatus.latest_round.all_factors || []),
              all_factors: runningRDAgentCandidates.length
                ? runningRDAgentCandidates
                : (autoCampaignStatus.latest_round.all_factors || autoCampaignStatus.latest_round.candidates || []),
              retained_count:
                autoCampaignStatus.latest_round.retained_count ||
                runningRDAgentCandidates.filter((factor: any) => factor.status === "accepted").length,
            }
          : null;
        return (
          <div>
            <Alert
              type="info"
              showIcon
              message={activeTab === "rdagent" ? "正在执行 RDAgent 挖掘" : "正在执行全自动化挖掘"}
              description={
                <Space direction="vertical" size={4}>
                  {activeTab === "rdagent" ? <span>当前阶段：{getRDAgentStageLabel((autoCampaignStatus as any).upstream_status)}</span> : null}
                  <span>当前轮次：第 {autoCampaignStatus.current_round} / {autoCampaignStatus.total_rounds || "-"} 轮</span>
                  <span>当前候选：第 {autoCampaignStatus.current_generation} / {autoCampaignStatus.total_generations || "-"} 个</span>
                  <span>累计保留：{autoCampaignStatus.retained_count || 0} 个因子</span>
                  <span>已用时间：{formatElapsedTime(elapsedTime)}</span>
                </Space>
              }
              style={{ marginBottom: 24 }}
            />
            {activeTab === "rdagent" ? renderRDAgentLoopOverview((autoCampaignStatus as any).upstream_status) : null}
            <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
              <Col xs={24} sm={8}><Card size="small"><div className="stat-label">总进度</div><div className="stat-value">{progressPercent}%</div></Card></Col>
              <Col xs={24} sm={8}><Card size="small"><div className="stat-label">当前最优分数</div><div className="stat-value">{autoCampaignStatus.best_fitness?.toFixed?.(1) || "0.0"}</div></Card></Col>
              <Col xs={24} sm={8}><Card size="small"><div className="stat-label">累计保留因子</div><div className="stat-value">{autoCampaignStatus.retained_count || 0}</div></Card></Col>
            </Row>
            <Progress percent={progressPercent} status="active" strokeColor="#3b82f6" />
            <div className="chart-section" style={{ marginTop: 24 }}>
              <h4 className="chart-title">{activeTab === "rdagent" ? "RDAgent 研究曲线" : "自动化研究曲线"}</h4>
              <div ref={progressChartRef} className="chart-container" style={{ height: 300 }} />
            </div>
            {activeTab === "rdagent" && (runningRDAgentRound || runningRDAgentCandidates.length) ? (
              <div style={{ marginTop: 24 }}>
                <Divider />
                <h3 className="result-title">当前 RDAgent 轮次证据</h3>
                {runningRDAgentRound
                  ? renderRDAgentTrace([runningRDAgentRound as RDAgentTraceRound])
                  : (
                    <div className="rdagent-candidate-grid">
                      {runningRDAgentCandidates.map((factor: any, index: number) => {
                        const score = factor.task_details?.rdagent?.candidate_score || {};
                        const diagnostics = getRDAgentCandidateDiagnostics(factor);
                        return (
                          <div key={`${factor.candidate_id || factor.name}-${index}`} className={`rdagent-candidate rdagent-candidate-${factor.status || "computed"}`}>
                            <div className="rdagent-candidate-top">
                              <span>{factor.name || `Candidate ${index + 1}`}</span>
                              <Tag>{factor.status || "computed"}</Tag>
                            </div>
                            <div className="factor-expression">{factor.expression}</div>
                            <Space wrap size={4}>
                              <Tag>rankIC {Number(score.rank_ic ?? factor.backtest_summary?.rank_ic_mean ?? 0).toFixed(3)}</Tag>
                              <Tag>Sharpe {Number(score.sharpe ?? factor.report_metrics?.sharpe ?? 0).toFixed(2)}</Tag>
                              <Tag>Coverage {Number(score.valid_coverage ?? 0).toFixed(2)}</Tag>
                            </Space>
                            {diagnostics.length ? (
                              <div className="rdagent-candidate-diagnostics">
                                {diagnostics.map((item, diagIndex) => (
                                  <Alert
                                    key={`${factor.candidate_id || factor.name}-running-diagnostic-${diagIndex}`}
                                    type={item.type as any}
                                    showIcon
                                    message={item.label}
                                    description={item.text}
                                  />
                                ))}
                              </div>
                            ) : null}
                          </div>
                        );
                      })}
                    </div>
                  )}
              </div>
            ) : null}
          </div>
        );
      }

      if (autoCampaignResult) {
        const isRDAgentResult = activeTab === "rdagent";
        const rdagentRounds = (autoCampaignResult.rounds || []) as unknown as RDAgentTraceRound[];
        const latestRDAgentRound = rdagentRounds[rdagentRounds.length - 1];
        const rdagentManualReport = latestRDAgentRound?.manual_report || autoCampaignResult.manual_report;
        const rdagentContinueRequest = latestRDAgentRound?.continue_mining_request || autoCampaignResult.continue_mining_request;
        return (
          <div className="result-shell">
            <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
              <Col xs={24} sm={8}><Card size="small"><div className="stat-label">累计最佳分数</div><div className="stat-value">{autoCampaignResult.best_score?.toFixed?.(1) || "0.0"}</div></Card></Col>
              <Col xs={24} sm={8}><Card size="small"><div className="stat-label">保留因子数</div><div className="stat-value">{autoCampaignResult.retained_factors?.length || 0}</div></Card></Col>
              <Col xs={24} sm={8}><Card size="small"><div className="stat-label">探索轮次</div><div className="stat-value">{autoCampaignResult.rounds?.length || 0}</div></Card></Col>
            </Row>

            {isRDAgentResult ? (
              <>
                {renderRDAgentLoopOverview("trace")}
              </>
            ) : null}

            <ResultSection
              kicker={isRDAgentResult ? "运行概览" : "结果概览"}
              title={activeTab === "rdagent" ? "RDAgent 研究结果" : "自动化研究结果"}
              description={isRDAgentResult ? "先看当前轮次和候选结论，完整证据放在下面的标签页里。" : "这里先展示最终产出和研究曲线，轮次细节按需展开。"}
            >
              <div className="chart-section" style={{ marginBottom: 0 }}>
                <h4 className="chart-title">{activeTab === "rdagent" ? "RDAgent 完整研究曲线" : "自动化完整研究曲线"}</h4>
                <div ref={resultChartRef} className="chart-container" style={{ height: 300 }} />
              </div>
            </ResultSection>

            {isRDAgentResult ? (
              <ResultSection
                kicker="研究细节"
                title="RDAgent 证据与后续动作"
                description="这部分保留完整 trace、人工报告和继续挖掘入口，但默认按标签分组，避免页面像说明文档。"
              >
                <Tabs
                  className="result-detail-tabs"
                  items={[
                    {
                      key: "trace",
                      label: (
                        <Space size={6}>
                          <FileSearchOutlined />
                          <span>Trace</span>
                        </Space>
                      ),
                      children: renderRDAgentTrace(rdagentRounds),
                    },
                    {
                      key: "report",
                      label: (
                        <Space size={6}>
                          <OrderedListOutlined />
                          <span>人工报告</span>
                        </Space>
                      ),
                      children: (
                        <>
                          {renderRDAgentManualReport(rdagentManualReport)}
                          {renderRDAgentContinueAction(rdagentContinueRequest)}
                        </>
                      ),
                    },
                  ]}
                />
              </ResultSection>
            ) : null}

            {!isRDAgentResult && autoCampaignResult.rounds?.length ? (
              <ResultSection
                kicker="过程回放"
                title="每轮摘要"
                description="保留轮次演进信息，但收成面向操作的摘要卡，不让说明性文字压住主界面。"
              >
                <div className="factors-list">
                  {autoCampaignResult.rounds.map((round) => (
                    <Card key={round.task_id} className="factor-card" size="small">
                      <div className="factor-header">
                        <div className="factor-info">
                          <Space wrap>
                            <Tag color="blue">第 {round.round_index} 轮</Tag>
                            <Tag color="purple">保留 {round.retained_count}</Tag>
                            <Tag color={round.factor_update_mode === "reselect" ? "geekblue" : round.factor_update_mode === "append" ? "gold" : "default"}>
                              {round.factor_update_mode === "reselect" ? "重选因子" : round.factor_update_mode === "append" ? "追加因子" : "初始轮"}
                            </Tag>
                          </Space>
                          <div className="factor-expression">基础因子：{(round.input_base_factors || []).join("，") || "—"}</div>
                        </div>
                        <div className="factor-stats">
                          <div className="stat-row"><span className="stat-label">Best:</span><span className="stat-value positive">{round.best_score?.toFixed?.(1) || "0.0"}</span></div>
                          <div className="stat-row"><span className="stat-label">Avg:</span><span className="stat-value">{round.avg_score?.toFixed?.(1) || "0.0"}</span></div>
                        </div>
                      </div>
                      {round.round_index > 1 ? (
                        <div style={{ marginTop: 12 }}>
                          {round.continuation_hypothesis?.hypothesis ? (
                            <Alert
                              type="info"
                              showIcon
                              style={{ marginBottom: 8 }}
                              message="本轮研究假设"
                              description={
                                <Space direction="vertical" size={4}>
                                  <span>{round.continuation_hypothesis.hypothesis}</span>
                                  {round.continuation_hypothesis.reason ? (
                                    <span style={{ color: "#475569" }}>原因：{round.continuation_hypothesis.reason}</span>
                                  ) : null}
                                  {round.continuation_hypothesis.target_goal ? (
                                    <span style={{ color: "#475569" }}>目标：{round.continuation_hypothesis.target_goal}</span>
                                  ) : null}
                                </Space>
                              }
                            />
                          ) : null}
                          {round.continuation_feedback?.hypothesis_evaluation ? (
                            <Alert
                              type={round.continuation_feedback.accepted_as_best ? "success" : "warning"}
                              showIcon
                              style={{ marginBottom: 8 }}
                              message={round.continuation_feedback.accepted_as_best ? "本轮已接管为新的 accepted best" : "本轮未接管 accepted best"}
                              description={
                                <Space direction="vertical" size={4}>
                                  <span>{round.continuation_feedback.hypothesis_evaluation}</span>
                                  {round.continuation_feedback.reason ? (
                                    <span style={{ color: "#475569" }}>{round.continuation_feedback.reason}</span>
                                  ) : null}
                                  {typeof round.continuation_feedback.score_delta === "number" ? (
                                    <span style={{ color: "#475569" }}>
                                      Score 变化：{round.continuation_feedback.score_delta >= 0 ? "+" : ""}
                                      {round.continuation_feedback.score_delta.toFixed(1)}
                                    </span>
                                  ) : null}
                                </Space>
                              }
                            />
                          ) : null}
                          <div style={{ marginBottom: 8, color: "#475569", fontSize: 13 }}>
                            上一轮基础因子：{(round.previous_base_factors || []).join("，") || "—"}
                          </div>
                          {(round.factor_changes?.added?.length || round.factor_changes?.removed?.length) ? (
                            <Space wrap style={{ marginBottom: 8 }}>
                              {(round.factor_changes?.added || []).map((name) => (
                                <Tag key={`added-${round.task_id}-${name}`} color="green">新增：{name}</Tag>
                              ))}
                              {(round.factor_changes?.removed || []).map((name) => (
                                <Tag key={`removed-${round.task_id}-${name}`} color="red">替换掉：{name}</Tag>
                              ))}
                            </Space>
                          ) : (
                            <div style={{ marginBottom: 8, color: "#64748b", fontSize: 13 }}>本轮基础因子未发生变化。</div>
                          )}
                          {round.selection_rationale ? (
                            <Alert
                              type="info"
                              showIcon
                              style={{ marginBottom: 8 }}
                              message="本轮选择理由"
                              description={round.selection_rationale}
                            />
                          ) : null}
                          {(round.selected_factors || []).length ? (
                            <div style={{ marginTop: 8 }}>
                              <div style={{ marginBottom: 6, color: "#334155", fontWeight: 500 }}>本轮 LLM 选择的因子</div>
                              <Space wrap>
                                {(round.selected_factors || []).map((name) => (
                                  <Tag key={`selected-${round.task_id}-${name}`} color="cyan">{name}</Tag>
                                ))}
                              </Space>
                              {(round.selected_factors || []).some((name) => round.per_factor_reason?.[name]) ? (
                                <div style={{ marginTop: 8 }}>
                                  {(round.selected_factors || []).map((name) => (
                                    round.per_factor_reason?.[name] ? (
                                      <div key={`reason-${round.task_id}-${name}`} style={{ color: "#475569", fontSize: 13, marginBottom: 4 }}>
                                        {name}：{round.per_factor_reason[name]}
                                      </div>
                                    ) : null
                                  ))}
                                </div>
                              ) : null}
                            </div>
                          ) : null}
                        </div>
                      ) : null}
                    </Card>
                  ))}
                </div>
              </ResultSection>
            ) : null}

            <ResultSection
              kicker={isRDAgentResult ? "人工确认区" : "最终产出"}
              title={isRDAgentResult ? "候选因子（需人工确认）" : "保留的因子"}
              description={isRDAgentResult ? "这里只放需要你判断和保存的候选，操作优先，证据详情按需展开。" : "最终保留下来的因子集中在这一组，报告和任务详情都按需展开。"}
              extra={isRDAgentResult ? <Tag color="gold">配置页视图</Tag> : <Tag color="blue">结果清单</Tag>}
            >
              {!autoCampaignResult.retained_factors?.length ? (
                <Alert message={isRDAgentResult ? "本轮没有候选因子达到待确认条件" : "没有因子满足当前筛选条件"} type="info" showIcon />
              ) : (
                <div className="factors-list">
                  {autoCampaignResult.retained_factors.map((factor: any, index: number) => {
                    const meta = factor.automation_meta || {};
                    const detailKey = getFactorDetailKey(factor, index);
                    const detailsExpanded = expandedDetailsKey === detailKey;
                    const factorTaskDetails = buildFactorTaskDetailsForDashboard(factor);
                    const factorReportUrl = factorTaskDetails?.report_url || factor.report_url;
                    return (
                      <Card key={`${factor.expression || factor.name || index}-${index}`} className="factor-card" size="small">
                        <div className="factor-header">
                          <div className="factor-info">
                            <Space wrap>
                              <Tag color="blue">Top {index + 1}</Tag>
                              <Tag color="cyan">第 {meta.round_index || "-"} 轮</Tag>
                              {isRDAgentResult ? <Tag color="gold">待人工确认</Tag> : null}
                              <span className="factor-name">{factor.name || `Factor_${index + 1}`}</span>
                              {factor.grade ? <Tag color="purple">{factor.grade}</Tag> : null}
                            </Space>
                            <div className="factor-expression">{factor.expression}</div>
                          </div>
                          <div className="factor-stats">
                            <div className="stat-row"><span className="stat-label">Score:</span><span className="stat-value positive">{factor.score?.toFixed?.(1) || "0.0"}</span></div>
                            <div className="stat-row"><span className="stat-label">Sharpe:</span><span className="stat-value">{factor.backtest_summary?.long_short_sharpe?.toFixed?.(2) || "-"}</span></div>
                            <div className="stat-row"><span className="stat-label">WQ Rating:</span><span className="stat-value">{factor.wq_brain?.wq_rating || "未同步"}</span></div>
                          </div>
                        </div>
                        <div className="factor-actions">
                          <Space wrap>
                            <Button type="primary" size="small" icon={<SaveOutlined />} onClick={() => saveFactor(factor, index)}>
                              {isRDAgentResult ? "确认并保存" : "保存到因子库"}
                            </Button>
                            <Button size="small" icon={<SettingOutlined />} onClick={() => toggleDetailsPreview(factor, index)}>
                              {detailsExpanded ? "收起详情" : "展开详情"}
                            </Button>
                            {factorReportUrl ? (
                              <Button size="small" icon={<BarChartOutlined />} onClick={() => toggleReportPreview(factorReportUrl)}>
                                {expandedReportUrl === resolveReportUrl(factorReportUrl) ? "收起报告" : "查看报告"}
                              </Button>
                            ) : null}
                          </Space>
                        </div>
                        {detailsExpanded ? (
                          <div id={`factor-details-${encodeURIComponent(detailKey)}`} style={{ marginTop: 16 }}>
                            <QuantTaskDetailsPanel
                              taskId={meta.round_task_id}
                              source={factor.source}
                              details={factorTaskDetails}
                            />
                          </div>
                        ) : null}
                        {factorReportUrl && expandedReportUrl === resolveReportUrl(factorReportUrl) ? (
                          <div id={`report-preview-${encodeURIComponent(resolveReportUrl(factorReportUrl))}`} style={{ marginTop: 16, border: "1px solid #e5e7eb", borderRadius: 8, overflow: "hidden", background: "#fff" }}>
                            <div style={{ padding: "10px 12px", borderBottom: "1px solid #e5e7eb", fontSize: 13, color: "#64748b" }}>
                              报告预览
                            </div>
                            <iframe
                              src={resolveReportUrl(factorReportUrl)}
                              title={`retained-report-preview-${index}`}
                              style={{ width: "100%", height: 720, border: "none", display: "block", background: "#fff" }}
                            />
                          </div>
                        ) : null}
                      </Card>
                    );
                  })}
                </div>
              )}
            </ResultSection>
          </div>
        );
      }
    }

    if (!mining && !currentStatus && !currentResult) {
      return (
        <div style={{ padding: 48, textAlign: "center" }}>
          <RocketOutlined style={{ fontSize: 48, color: "#94a3b8", marginBottom: 16 }} />
          <h3 style={{ marginBottom: 8 }}>准备开始{activeTab === "manual" ? "手动挖掘" : activeTab === "rdagent" ? "RDAgent 挖掘" : "自动挖掘"}</h3>
          <p style={{ color: "#64748b" }}>
            {activeTab === "manual"
              ? "选择基础因子后运行 FactorHub 原有挖掘逻辑。"
              : activeTab === "rdagent"
                ? "配置研究目标和筛选条件后，在 FactorHub 内运行多轮候选生成、回测与保留。"
                : "从因子库导入基础因子后，通过 FactorHub 本地自动挖掘链路完成表达式生成、回测评估与候选迭代。"}
          </p>
        </div>
      );
    }

    if (mining) {
      const currentGeneration = currentStatus?.current_generation ?? 0;
      const totalGenerations = currentStatus?.total_generations ?? 0;
      const bestFitness = currentStatus?.best_fitness ?? 0;
      const avgFitness = currentStatus?.avg_fitness ?? 0;
      const progressPercent = totalGenerations > 0 ? Math.round((currentGeneration / totalGenerations) * 100) : 0;
      const runningManualCandidates = activeTab === "manual" ? (manualStatus?.candidates || []) : [];
      const runningAutoCandidates = isAutoLikeTab ? ((autoStatus as MiningStatus | null)?.candidates || []) : [];
      const seedResult = isAutoLikeTab ? autoSeedState?.result || null : null;

      return (
        <div>
          {seedResult ? (
            <div style={{ marginBottom: 24 }}>
              <Alert
                type="info"
                showIcon
                message="继续探索已启动"
                description="上一轮研究结果已保留，新的研究曲线会在候选因子逐步完成后持续更新。"
              />
            </div>
          ) : null}

          {seedResult ? (
            <div style={{ marginBottom: 24 }}>
              <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
                <Col xs={24} sm={8}><Card size="small"><div className="stat-label">上一轮最佳分数</div><div className="stat-value">{seedResult.best_score?.toFixed?.(1) || "0.0"}</div></Card></Col>
                <Col xs={24} sm={8}><Card size="small"><div className="stat-label">上一轮平均分数</div><div className="stat-value">{seedResult.avg_score?.toFixed?.(1) || "0.0"}</div></Card></Col>
                <Col xs={24} sm={8}><Card size="small"><div className="stat-label">上一轮发现因子数</div><div className="stat-value">{seedResult.factors?.length || 0}</div></Card></Col>
              </Row>
              <div className="chart-section" style={{ marginBottom: 24 }}>
                <h4 className="chart-title">上一轮完整研究曲线</h4>
                <div ref={resultChartRef} className="chart-container" style={{ height: 300 }} />
              </div>
            </div>
          ) : null}

          <div style={{ marginBottom: 24 }}>
            <Alert
              type="info"
              showIcon
              message={activeTab === "manual" ? "正在执行手动挖掘" : activeTab === "rdagent" ? "正在执行 RDAgent 挖掘" : "正在执行自动挖掘"}
              description={
                <Space direction="vertical" size={4}>
                  <span>当前进度：第 {currentGeneration} / {totalGenerations || "-"} 轮</span>
                  <span>已用时间：{formatElapsedTime(elapsedTime)}</span>
                </Space>
              }
            />
          </div>

          <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
            <Col xs={24} sm={8}><Card size="small"><div className="stat-label">进度</div><div className="stat-value">{progressPercent}%</div></Card></Col>
            <Col xs={24} sm={8}><Card size="small"><div className="stat-label">最优值</div><div className="stat-value">{bestFitness?.toFixed?.(4) || "0.0000"}</div></Card></Col>
            <Col xs={24} sm={8}><Card size="small"><div className="stat-label">平均值</div><div className="stat-value">{avgFitness?.toFixed?.(4) || "0.0000"}</div></Card></Col>
          </Row>

          <Progress percent={progressPercent} status="active" strokeColor="#3b82f6" />

          <div className="chart-section" style={{ marginTop: 24 }}>
            <h4 className="chart-title">{activeTab === "manual" ? "进化曲线" : activeTab === "rdagent" ? "RDAgent 研究曲线" : "研究曲线"}</h4>
            <div ref={progressChartRef} className="chart-container" style={{ height: 300 }} />
          </div>
          {activeTab === "manual" && runningManualCandidates.length ? (
            <div style={{ marginTop: 24 }}>
              <Divider />
              <h3 className="result-title">当前最优候选</h3>
              <div className="factors-list">
                {runningManualCandidates.map((factor: any, index: number) => (
                  <Card key={`${factor.expression || factor.name || index}-${index}`} className="factor-card" size="small">
                    <div className="factor-header">
                      <div className="factor-info">
                        <Space>
                          <Tag color="blue">实时 {index + 1}</Tag>
                          <span className="factor-name">{factor.name || `Mined_Factor_${index + 1}`}</span>
                        </Space>
                        <div className="factor-expression">{factor.expression}</div>
                      </div>
                      <div className="factor-stats">
                        <div className="stat-row"><span className="stat-label">Fitness:</span><span className="stat-value positive">{factor.fitness?.toFixed?.(4) || "0.0000"}</span></div>
                        <div className="stat-row"><span className="stat-label">IC:</span><span className={`stat-value ${Number(factor.ic || 0) > 0 ? "positive" : "negative"}`}>{factor.ic?.toFixed?.(4) || "0.0000"}</span></div>
                        <div className="stat-row"><span className="stat-label">IR:</span><span className={`stat-value ${Number(factor.ir || 0) > 0 ? "positive" : "negative"}`}>{factor.ir?.toFixed?.(4) || "0.0000"}</span></div>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            </div>
          ) : null}
          {isAutoLikeTab && runningAutoCandidates.length ? (
            <div style={{ marginTop: 24 }}>
              <Divider />
              <h3 className="result-title">已完成候选因子</h3>
              <div className="factors-list">
                {runningAutoCandidates.map((factor: any, index: number) => (
                  <Card key={`${factor.expression || factor.name || index}-${index}`} className="factor-card" size="small">
                    <div className="factor-header">
                      <div className="factor-info">
                        <Space>
                          <Tag color="blue">实时 {index + 1}</Tag>
                          <span className="factor-name">{factor.name || `Candidate_${index + 1}`}</span>
                          {factor.grade ? <Tag color="purple">{factor.grade}</Tag> : null}
                        </Space>
                        <div className="factor-expression">{factor.expression}</div>
                      </div>
                      <div className="factor-stats">
                        <div className="stat-row"><span className="stat-label">Score:</span><span className="stat-value positive">{factor.score?.toFixed?.(1) || "0.0"}</span></div>
                        <div className="stat-row"><span className="stat-label">Sharpe:</span><span className="stat-value">{factor.report_metrics?.sharpe?.toFixed?.(2) || "-"}</span></div>
                        <div className="stat-row"><span className="stat-label">CAGR:</span><span className="stat-value">{factor.report_metrics?.cagr?.toFixed?.(2) || "-"}</span></div>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      );
    }

    if (currentResult) {
      const factorsToRender = (currentResult as any).factors || [];
      const roundEvaluation = isAutoLikeTab ? getRoundEvaluation(currentResult as AutoMiningResult) : null;
      return (
        <div>
          <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
            <Col xs={24} sm={8}><Card size="small"><div className="stat-label">{activeTab === "manual" ? "最佳适应度" : "最佳分数"}</div><div className="stat-value">{activeTab === "manual" ? (manualResult?.best_fitness?.toFixed?.(4) || "0.0000") : (autoResult?.best_score?.toFixed?.(1) || "0.0")}</div></Card></Col>
            <Col xs={24} sm={8}><Card size="small"><div className="stat-label">{activeTab === "manual" ? "平均适应度" : "平均分数"}</div><div className="stat-value">{activeTab === "manual" ? (manualResult?.avg_fitness?.toFixed?.(4) || "0.0000") : (autoResult?.avg_score?.toFixed?.(1) || "0.0")}</div></Card></Col>
            <Col xs={24} sm={8}><Card size="small"><div className="stat-label">发现因子数</div><div className="stat-value">{factorsToRender.length}</div></Card></Col>
          </Row>

          <div className="chart-section" style={{ marginBottom: 24 }}>
            <h4 className="chart-title">{activeTab === "manual" ? "完整进化曲线" : "完整研究曲线"}</h4>
            <div ref={resultChartRef} className="chart-container" style={{ height: 300 }} />
          </div>

          {isAutoLikeTab && roundEvaluation ? (
            <Card size="small" title="本轮评估报告" style={{ marginBottom: 24, background: "#fafafa" }}>
              <Row gutter={[16, 16]}>
                <Col xs={24} md={12}>
                  <div style={{ fontSize: 12, color: "#64748b", marginBottom: 6 }}>主要问题</div>
                  <div style={{ color: "#0f172a" }}>{roundEvaluation.primary_problem || "暂无"}</div>
                </Col>
                <Col xs={24} md={12}>
                  <div style={{ fontSize: 12, color: "#64748b", marginBottom: 6 }}>建议优化方向</div>
                  <Tag color="blue" style={{ fontSize: 13 }}>{roundEvaluation.recommended_goal || "暂无"}</Tag>
                </Col>
                {(roundEvaluation.base_factors || []).length ? (
                  <Col span={24}>
                    <div style={{ fontSize: 12, color: "#64748b", marginBottom: 6 }}>当前基础因子</div>
                    <Space wrap>
                      {(roundEvaluation.base_factors || []).map((name: string) => (
                        <Tag key={`round-evaluation-base-${name}`}>{name}</Tag>
                      ))}
                    </Space>
                  </Col>
                ) : null}
                {(roundEvaluation.suggested_actions || []).length ? (
                  <Col span={24}>
                    <div style={{ fontSize: 12, color: "#64748b", marginBottom: 6 }}>建议动作</div>
                    <Space direction="vertical" size={4} style={{ width: "100%" }}>
                      {(roundEvaluation.suggested_actions || []).map((item: string, idx: number) => (
                        <span key={`round-evaluation-action-${idx}`}>• {item}</span>
                      ))}
                    </Space>
                  </Col>
                ) : null}
                {roundEvaluation.metric_snapshot ? (
                  <Col span={24}>
                    <div style={{ fontSize: 12, color: "#64748b", marginBottom: 6 }}>关键指标快照</div>
                    <Space wrap>
                      {roundEvaluation.metric_snapshot.score !== undefined ? <Tag color="green">Score：{Number(roundEvaluation.metric_snapshot.score || 0).toFixed(1)}</Tag> : null}
                      {roundEvaluation.metric_snapshot.report_sharpe !== undefined && roundEvaluation.metric_snapshot.report_sharpe !== null ? <Tag>Report Sharpe：{Number(roundEvaluation.metric_snapshot.report_sharpe).toFixed(2)}</Tag> : null}
                      {roundEvaluation.metric_snapshot.ls_sharpe !== undefined && roundEvaluation.metric_snapshot.ls_sharpe !== null ? <Tag>L/S Sharpe：{Number(roundEvaluation.metric_snapshot.ls_sharpe).toFixed(2)}</Tag> : null}
                      {roundEvaluation.metric_snapshot.report_max_drawdown !== undefined && roundEvaluation.metric_snapshot.report_max_drawdown !== null ? <Tag color="orange">Max Drawdown：{Number(roundEvaluation.metric_snapshot.report_max_drawdown).toFixed(2)}</Tag> : null}
                      {roundEvaluation.metric_snapshot.ls_return !== undefined && roundEvaluation.metric_snapshot.ls_return !== null ? <Tag>L/S Return：{Number(roundEvaluation.metric_snapshot.ls_return).toFixed(2)}</Tag> : null}
                    </Space>
                  </Col>
                ) : null}
              </Row>
            </Card>
          ) : null}

          <Divider />
          <h3 className="result-title">发现的因子</h3>

          {!factorsToRender.length ? (
            <Alert message="未发现符合条件的因子" type="info" showIcon style={{ marginTop: 16 }} />
          ) : (
            <div className="factors-list">
              {factorsToRender.map((factor: any, index: number) => {
                const detailKey = getFactorDetailKey(factor, index);
                const detailsExpanded = expandedDetailsKey === detailKey;
                const factorTaskDetails = buildFactorTaskDetailsForDashboard(factor);
                const factorReportUrl = factorTaskDetails?.report_url || factor.report_url;
                return (
                <Card key={index} className="factor-card" size="small">
                  <div className="factor-header">
                    <div className="factor-info">
                      <Space>
                        <Tag color="blue">Top {index + 1}</Tag>
                        <span className="factor-name">{factor.name || `Factor_${index + 1}`}</span>
                        {isAutoLikeTab && factor.grade ? <Tag color="purple">{factor.grade}</Tag> : null}
                        {isAutoLikeTab && factor.source ? <Tag color="cyan">{factor.source}</Tag> : null}
                      </Space>
                      <div className="factor-expression">{factor.expression}</div>
                    </div>
                    <div className="factor-stats">
                      {activeTab === "manual" ? (
                        <>
                          <div className="stat-row"><span className="stat-label">IC:</span><span className={`stat-value ${factor.ic > 0 ? "positive" : "negative"}`}>{factor.ic?.toFixed?.(4)}</span></div>
                          <div className="stat-row"><span className="stat-label">IR:</span><span className={`stat-value ${factor.ir > 0 ? "positive" : "negative"}`}>{factor.ir?.toFixed?.(4)}</span></div>
                        </>
                      ) : (
                        <>
                          <div className="stat-row"><span className="stat-label">Score:</span><span className="stat-value positive">{factor.score?.toFixed?.(1) || "0.0"}</span></div>
                          <div className="stat-row"><span className="stat-label">Sharpe:</span><span className="stat-value">{factor.report_metrics?.sharpe?.toFixed?.(2) || "-"}</span></div>
                          <div className="stat-row"><span className="stat-label">CAGR:</span><span className="stat-value">{factor.report_metrics?.cagr?.toFixed?.(2) || "-"}</span></div>
                        </>
                      )}
                    </div>
                  </div>
                  <div className="factor-actions">
                    <Space>
                      <Button type="primary" size="small" icon={<SaveOutlined />} onClick={() => saveFactor(factor, index)}>
                        保存到因子库
                      </Button>
                      {isAutoLikeTab ? (
                        <Button size="small" onClick={() => toggleDetailsPreview(factor, index)}>
                          {detailsExpanded ? "收起详情" : "展开详情"}
                        </Button>
                      ) : null}
                      {isAutoLikeTab && factorReportUrl ? (
                        <Button size="small" icon={<BarChartOutlined />} onClick={() => toggleReportPreview(factorReportUrl)}>
                          {expandedReportUrl === resolveReportUrl(factorReportUrl) ? "收起报告" : "查看报告"}
                        </Button>
                      ) : null}
                    </Space>
                  </div>
                  {isAutoLikeTab && detailsExpanded ? (
                    <div id={`factor-details-${encodeURIComponent(detailKey)}`} style={{ marginTop: 16 }}>
                      <QuantTaskDetailsPanel
                        taskId={factor.task_id}
                        source={factor.source}
                        details={factorTaskDetails}
                      />
                    </div>
                  ) : null}
                  {isAutoLikeTab && factorReportUrl && expandedReportUrl === resolveReportUrl(factorReportUrl) ? (
                    <div id={`report-preview-${encodeURIComponent(resolveReportUrl(factorReportUrl))}`} style={{ marginTop: 16, border: "1px solid #e5e7eb", borderRadius: 8, overflow: "hidden", background: "#fff" }}>
                      <div style={{ padding: "10px 12px", borderBottom: "1px solid #e5e7eb", fontSize: 13, color: "#64748b" }}>
                        报告预览
                      </div>
                      <iframe
                        src={resolveReportUrl(factorReportUrl)}
                        title={`report-preview-${index}`}
                        style={{ width: "100%", height: 720, border: "none", display: "block", background: "#fff" }}
                      />
                    </div>
                  ) : null}
                </Card>
                );
              })}
            </div>
          )}

          <div className="result-actions" style={{ marginTop: 24 }}>
            <Space wrap>
              <Button type="primary" icon={<SaveOutlined />} onClick={saveAllFactors}>
                {activeTab === "rdagent" ? "全部确认并保存" : "全部保存到因子库"}
              </Button>
              {activeTab === "auto" && autoStatus?.task_id ? (
                <Button onClick={() => {
                  setShowContinuePanel((prev) => !prev);
                  continueForm.setFieldsValue({
                    prompt: autoForm.getFieldValue("prompt"),
                    direction: autoForm.getFieldValue("direction"),
                    factor_update_mode: "append",
                    auto_additional_factor_count: 5,
                    n_candidates: autoForm.getFieldValue("n_candidates"),
                    n_groups: autoForm.getFieldValue("n_groups"),
                    holding_period: autoForm.getFieldValue("holding_period"),
                    neutralize_industry: autoForm.getFieldValue("neutralize_industry"),
                    neutralize_cap: autoForm.getFieldValue("neutralize_cap"),
                    additional_base_factors: [],
                  });
                }}>
                  {showContinuePanel ? "收起继续探索" : "继续探索 / 继续优化"}
                </Button>
              ) : null}
            </Space>
          </div>
          {activeTab === "auto" && showContinuePanel ? (
            <Card size="small" style={{ marginTop: 16 }} title="继续探索">
              <Form form={continueForm} layout="vertical" onFinish={startContinueAutoMining}>
                <Tag color="blue" style={{ marginBottom: 16 }}>基于上一轮最优表达式继续优化，可追加新的基础因子</Tag>
                {continuationInsightSummary ? (
                  <Card size="small" style={{ marginBottom: 16, background: "#fafafa" }} title="上一轮诊断">
                    <Space direction="vertical" size={10} style={{ width: "100%" }}>
                      <div>
                        <div style={{ fontSize: 12, color: "#64748b", marginBottom: 6 }}>原有基础因子</div>
                        {continuationInsightSummary.baseFactors.length ? (
                          <Space wrap>
                            {continuationInsightSummary.baseFactors.map((factorName) => (
                              <Tag key={factorName}>{factorName}</Tag>
                            ))}
                          </Space>
                        ) : (
                          <span style={{ color: "#94a3b8" }}>暂无</span>
                        )}
                      </div>
                      <div>
                        <div style={{ fontSize: 12, color: "#64748b", marginBottom: 6 }}>当前短板 / 风险</div>
                        {continuationInsightSummary.weaknesses.length ? (
                          <Space direction="vertical" size={4}>
                            {continuationInsightSummary.weaknesses.map((item, idx) => (
                              <span key={`${idx}-${item}`}>• {item}</span>
                            ))}
                          </Space>
                        ) : (
                          <span style={{ color: "#94a3b8" }}>系统暂未提取到明确短板，LLM 会结合报告指标继续判断</span>
                        )}
                      </div>
                      <div>
                        <div style={{ fontSize: 12, color: "#64748b", marginBottom: 6 }}>建议优化方向</div>
                        {continuationInsightSummary.optimizationDirections.length ? (
                          <Space direction="vertical" size={4}>
                            {continuationInsightSummary.optimizationDirections.map((item, idx) => (
                              <span key={`${idx}-${item}`}>• {item}</span>
                            ))}
                          </Space>
                        ) : (
                          <span style={{ color: "#94a3b8" }}>系统暂未提取到明确建议，LLM 会基于上一轮报告生成优化方向</span>
                        )}
                      </div>
                    </Space>
                  </Card>
                ) : null}
                <Form.Item label="因子更新方式" name="factor_update_mode" initialValue="append">
                  <Select>
                    <Option value="append">追加新因子</Option>
                    <Option value="reselect">根据上一轮结果重新选择因子</Option>
                  </Select>
                </Form.Item>
                <Form.Item label="继续探索提示词" name="prompt">
                  <Input.TextArea rows={3} placeholder="例如：在上一轮结果基础上继续优化，优先提升 L/S Sharpe，并兼顾 WQ Fitness" />
                </Form.Item>
                <Form.Item label="优化方向" name="direction">
                  <Select placeholder="选择本轮唯一主优化指标" options={OPTIMIZATION_DIRECTION_OPTIONS} allowClear />
                </Form.Item>
                <Row gutter={16}>
                  <Col xs={24} sm={12}>
                    <Form.Item label="LLM 选择因子数" name="auto_additional_factor_count" tooltip="追加模式下表示新增因子数；重选模式下表示下一轮基础因子总数上限">
                      <InputNumber min={1} max={10} style={{ width: "100%" }} />
                    </Form.Item>
                  </Col>
                  <Col xs={24} sm={12} style={{ display: "flex", alignItems: "end" }}>
                    <Form.Item style={{ width: "100%", marginBottom: 24 }}>
                      <Button block onClick={() => void handleContinueLLMSelectFactors()} loading={continueSelectingFactors}>
                        LLM 根据上一轮报告选择因子
                      </Button>
                    </Form.Item>
                  </Col>
                </Row>
                {continueSelectionSummary ? (
                  <Alert
                    type="info"
                    showIcon
                    style={{ marginBottom: 16 }}
                    message="LLM 已补充下一轮因子"
                    description={continueSelectionSummary}
                  />
                ) : null}
                <Form.Item label="下一轮基础因子" name="additional_base_factors" tooltip="追加模式下表示追加的新因子；重选模式下表示重新选择后的完整基础因子列表">
                  <Select mode="multiple" placeholder="从因子库选择下一轮基础因子" showSearch optionLabelProp="label" maxTagCount="responsive">
                    {renderFactorOptions()}
                  </Select>
                </Form.Item>
                <Row gutter={16}>
                  <Col xs={24} sm={8}><Form.Item label="候选轮次" name="n_candidates"><InputNumber min={1} max={10} style={{ width: "100%" }} /></Form.Item></Col>
                  <Col xs={24} sm={8}><Form.Item label="分组数" name="n_groups"><InputNumber min={2} max={20} style={{ width: "100%" }} /></Form.Item></Col>
                  <Col xs={24} sm={8}><Form.Item label="持有期" name="holding_period"><InputNumber min={1} max={60} style={{ width: "100%" }} /></Form.Item></Col>
                </Row>
                <Row gutter={16}>
                  <Col xs={24} sm={12}>
                    <Form.Item name="neutralize_industry" valuePropName="checked" style={{ marginBottom: 12 }}>
                      <Checkbox>行业中性化</Checkbox>
                    </Form.Item>
                  </Col>
                  <Col xs={24} sm={12}>
                    <Form.Item name="neutralize_cap" valuePropName="checked" style={{ marginBottom: 12 }}>
                      <Checkbox>市值中性化</Checkbox>
                    </Form.Item>
                  </Col>
                </Row>
                <Space>
                  <Button type="primary" htmlType="submit" loading={continueExploring} disabled={mining}>
                    开始继续探索
                  </Button>
                  <Button onClick={() => setShowContinuePanel(false)}>取消</Button>
                </Space>
              </Form>
            </Card>
          ) : null}
        </div>
      );
    }

    return renderReadyState(
      activeTab === "manual" ? "等待展示结果" : isAutoLikeTab ? "等待展示研究结果" : "等待展示结果",
      "当前页面还没有可展示的运行结果。"
    );
  };

  return (
    <div className="factor-mining-container">
      <div className="bg-gradient" />
      <div className="bg-grid" />
      <div className="factor-mining-content">
        <div className="page-header">
          <div className="header-content">
            <RocketOutlined className="header-icon" />
            <div>
              <h1 className="page-title">因子挖掘</h1>
              <p className="page-subtitle">统一因子库、统一回测、统一报告</p>
            </div>
          </div>
        </div>

        <Tabs activeKey={activeTab} onChange={(key) => setActiveTab(key as MiningMode)} items={[
          {
            key: "manual",
            label: "手动挖掘",
            children: (
              <Row gutter={[24, 24]}>
                <Col xs={24} lg={8}>
                  <Card title="遗传算法配置" className="config-card">
                    <Form form={manualForm} layout="vertical" onFinish={startManualMining}>
                      <Divider styles={{ content: { margin: 0 } }} titlePlacement="left">基础配置</Divider>
                      <Form.Item label="股票代码" name="stock_code" rules={[{ required: true, message: "请输入股票代码" }]}><Input placeholder="例如：000001、600000" /></Form.Item>
                      <Form.Item label="日期范围" name="dateRange" rules={[{ required: true, message: "请选择日期范围" }]}>
                        <RangePicker
                          style={{ width: "100%" }}
                          allowClear
                          inputReadOnly={isMobileView}
                          placement={isMobileView ? "bottomLeft" : undefined}
                          getPopupContainer={(trigger) => trigger.parentElement || document.body}
                        />
                      </Form.Item>

                      <Divider styles={{ content: { margin: 0 } }} titlePlacement="left">基础因子选择</Divider>
                      <p className="text-hint">选择作为遗传算法输入的基础因子（可搜索因子名称）</p>
                      <Form.Item label="LLM 选因子提示词" name="manual_prompt" rules={[{ required: true, message: "请输入 LLM 选因子提示词" }]}>
                        <Input.TextArea rows={3} placeholder="例如：为当前股票挑选一组更适合手动遗传挖掘的基础因子，优先保证稳定性和可解释性。" />
                      </Form.Item>
                      <Row gutter={16}>
                        <Col span={12}>
                          <Form.Item label="优化方向" name="manual_direction">
                            <Select placeholder="选择唯一主优化目标" options={OPTIMIZATION_DIRECTION_OPTIONS} allowClear />
                          </Form.Item>
                        </Col>
                        <Col span={12}>
                          <Form.Item label="最多因子数" name="manual_max_factor_count">
                            <InputNumber min={1} max={20} style={{ width: "100%" }} />
                          </Form.Item>
                        </Col>
                      </Row>
                      <Form.Item>
                        <Button onClick={() => void handleManualLLMSelectFactors()} loading={llmSelectingFactors}>
                          LLM 自动从因子库筛选基础因子
                        </Button>
                      </Form.Item>
                      {llmSelectionSummary ? <Alert type="info" showIcon style={{ marginBottom: 16 }} message="LLM 筛选说明" description={llmSelectionSummary} /> : null}
                      <Form.Item name="base_factors" rules={[{ required: true, message: "请至少选择一个基础因子" }]}>
                        <Select mode="multiple" placeholder="输入因子名称搜索" showSearch optionLabelProp="label" maxTagCount="responsive">{renderFactorOptions()}</Select>
                      </Form.Item>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                        <span className="text-hint">已选择 <strong style={{ color: "#3b82f6" }}>{selectedFactorCount}</strong> 个因子</span>
                        <Space size="small">
                          <Button type="link" size="small" onClick={() => manualForm.setFieldsValue({ base_factors: factors.map((f) => f.name) })}>全选</Button>
                          <Button type="link" size="small" onClick={() => manualForm.setFieldsValue({ base_factors: [] })}>清空</Button>
                        </Space>
                      </div>

                      <Divider styles={{ content: { margin: 0 } }} titlePlacement="left">算法参数</Divider>
                      <Row gutter={16}>
                        <Col span={12}><Form.Item label="种群大小" name="population_size"><InputNumber min={10} max={200} style={{ width: "100%" }} /></Form.Item></Col>
                        <Col span={12}><Form.Item label="迭代次数" name="n_generations"><InputNumber min={1} max={100} style={{ width: "100%" }} /></Form.Item></Col>
                      </Row>
                      <Row gutter={16}>
                        <Col span={12}><Form.Item label="变异率" name="mutation_rate"><InputNumber min={0} max={1} step={0.05} style={{ width: "100%" }} /></Form.Item></Col>
                        <Col span={12}><Form.Item label="交叉率" name="crossover_rate"><InputNumber min={0} max={1} step={0.05} style={{ width: "100%" }} /></Form.Item></Col>
                      </Row>
                      <Form.Item label="精英保留数量" name="elite_size"><InputNumber min={0} max={20} style={{ width: "100%" }} /></Form.Item>
                      <Divider styles={{ content: { margin: 0 } }} titlePlacement="left">适应度函数</Divider>
                      <Form.Item label="优化目标" name="fitness_objective"><Select><Option value="ic_mean">IC均值</Option><Option value="ir_ratio">IR比率</Option><Option value="sharpe">夏普比率</Option><Option value="combined">综合得分</Option></Select></Form.Item>
                      <Form.Item label="IC阈值" name="ic_threshold"><InputNumber min={0} step={0.01} style={{ width: "100%" }} /></Form.Item>
                      <Form.Item><Button type="primary" htmlType="submit" icon={<PlayCircleOutlined />} loading={loading && activeTab === "manual"} block size="large" disabled={mining}>{mining && activeTab === "manual" ? "挖掘中..." : "开始挖掘"}</Button></Form.Item>
                    </Form>
                  </Card>
                </Col>
                <Col xs={24} lg={16}><Card title="挖掘结果" className="result-card">{renderStatusCard()}</Card></Col>
              </Row>
            ),
          },
          {
            key: "auto",
            label: "自动挖掘",
            children: (
              <Row gutter={[24, 24]}>
                <Col xs={24} lg={8}>
                  <Card title="自动挖掘配置" className="config-card">
                    <Form form={autoForm} layout="vertical" onFinish={handleAutoMiningSubmit} onValuesChange={persistAutoMiningForm}>
                      {renderAutoModeSelector()}
                      {isMobileView ? (
                        <>
                          <Form.Item label="自动挖掘提示词" name="prompt" rules={[{ required: true, message: "请输入自动挖掘目标" }]}>
                            <Input.TextArea rows={4} placeholder="例如：从导入因子中自动挖掘更稳定的量价复合因子，优先提升 Sharpe 与稳定性" />
                          </Form.Item>
                          <Form.Item label="优化方向（可选）" name="direction">
                            <Select placeholder="选择本轮唯一主优化指标" options={OPTIMIZATION_DIRECTION_OPTIONS} allowClear />
                          </Form.Item>
                          <Form.Item label="最多因子数" name="max_factor_count" tooltip="LLM 会在这个上限内自主决定选择多少个基础因子，避免后续优化过于复杂">
                            <InputNumber min={1} max={50} style={{ width: "100%" }} />
                          </Form.Item>
                          <Form.Item>
                            <Button block onClick={() => void handleLLMSelectFactors()} loading={llmSelectingFactors}>LLM 自动从因子库筛选因子</Button>
                          </Form.Item>
                          {llmSelectionSummary ? <Alert type="info" showIcon style={{ marginBottom: 16 }} message="LLM 已完成因子筛选" description={llmSelectionSummary} /> : null}

                          <Form.Item name="base_factors" rules={[{ required: true, message: "请至少导入一个基础因子" }]}>
                            <Select mode="multiple" placeholder="选择要导入的因子" showSearch optionLabelProp="label" maxTagCount="responsive">{renderFactorOptions()}</Select>
                          </Form.Item>
                          <div className="mobile-import-bar">
                            <span className="text-hint">已导入 <strong style={{ color: "#3b82f6" }}>{selectedFactorCount}</strong> 个因子</span>
                            <Space size="small">
                              <Button type="link" size="small" onClick={() => updateAutoBaseFactors(factors.map((f) => f.name))}>全选</Button>
                              <Button type="link" size="small" onClick={() => updateAutoBaseFactors([])}>清空</Button>
                            </Space>
                          </div>

                          <Collapse bordered={false} className="mobile-config-collapse" defaultActiveKey={["params"]}>
                            <Panel header="研究参数" key="params">
                              <Form.Item label="日期范围" name="dateRange" rules={[{ required: true, message: "请选择日期范围" }]}>
                                <RangePicker
                                  style={{ width: "100%" }}
                                  allowClear
                                  inputReadOnly
                                  placement="bottomLeft"
                                  getPopupContainer={(trigger) => trigger.parentElement || document.body}
                                />
                              </Form.Item>
                              <Row gutter={16}>
                                <Col xs={24} sm={12}><Form.Item label="股票池" name="universe"><Select><Option value="hs300">HS300</Option><Option value="zz500">ZZ500</Option><Option value="zz1000">ZZ1000</Option><Option value="all">全市场</Option></Select></Form.Item></Col>
                                <Col xs={24} sm={12}><Form.Item label="基准" name="benchmark"><Select><Option value="hs300">HS300</Option><Option value="zz500">ZZ500</Option><Option value="zz1000">ZZ1000</Option></Select></Form.Item></Col>
                              </Row>
                              <Row gutter={16}>
                                <Col xs={24} sm={12}><Form.Item label="分组数" name="n_groups"><InputNumber min={2} max={20} style={{ width: "100%" }} /></Form.Item></Col>
                                <Col xs={24} sm={12}><Form.Item label="持有期" name="holding_period"><InputNumber min={1} max={60} style={{ width: "100%" }} /></Form.Item></Col>
                              </Row>
                              <Form.Item label="候选轮次" name="n_candidates" tooltip="对应 FactorHub 本地候选表达式筛选轮次"><InputNumber min={1} max={10} style={{ width: "100%" }} /></Form.Item>
                              <Row gutter={16}>
                                <Col xs={24} sm={12}>
                                  <Form.Item name="neutralize_industry" valuePropName="checked" style={{ marginBottom: 12 }}>
                                    <Checkbox>行业中性化</Checkbox>
                                  </Form.Item>
                                </Col>
                                <Col xs={24} sm={12}>
                                  <Form.Item name="neutralize_cap" valuePropName="checked" style={{ marginBottom: 12 }}>
                                    <Checkbox>市值中性化</Checkbox>
                                  </Form.Item>
                                </Col>
                              </Row>
                            </Panel>
                            <Panel header="LLM 配置" key="llm">
                              <Form form={llmForm} layout="vertical">
                                <Form.Item label="API Key" name="api_key">
                                  <Input.Password placeholder={llmConfig?.has_api_key ? `已配置：${llmConfig?.api_key_masked || ''}；留空则保持不变` : '填写 DeepSeek / OpenAI 兼容 API Key'} />
                                </Form.Item>
                                <Form.Item label="Base URL" name="base_url" rules={[{ required: true, message: '请输入 Base URL' }]}>
                                  <Input placeholder="https://api.deepseek.com/v1" />
                                </Form.Item>
                                <Form.Item label="模型" name="model" rules={[{ required: true, message: '请输入模型名' }]}>
                                  <Input placeholder="deepseek-chat" />
                                </Form.Item>
                                <Tag color={llmConfig?.has_api_key ? 'green' : 'orange'} style={{ marginBottom: 16 }}>
                                  {llmConfig?.has_api_key ? 'LLM 已连接' : 'LLM 未配置'}
                                </Tag>
                                <div className="mobile-action-stack">
                                  <Button type="primary" onClick={() => void saveLLMConfig()} loading={llmConfigSaving}>保存配置</Button>
                                  <Button onClick={() => void restartLLMService()} loading={llmRestarting}>检查服务状态</Button>
                                  <Button onClick={() => void loadLLMConfig()}>刷新状态</Button>
                                </div>
                              </Form>
                            </Panel>
                          </Collapse>

                          {autoLaunchMode === "campaign" ? (
                            <Card size="small" title="连续探索配置" style={{ marginBottom: 16 }}>
                              <Row gutter={16}>
                                <Col xs={24} sm={12}>
                                  <Form.Item label="累计探索次数" name="automation_exploration_rounds" rules={[{ required: true, message: "请输入累计探索次数" }]}>
                                    <InputNumber min={1} max={10} style={{ width: "100%" }} />
                                  </Form.Item>
                                </Col>
                                <Col xs={24} sm={12}>
                                  <Form.Item label="每轮候选轮次" name="automation_n_candidates_per_round" rules={[{ required: true, message: "请输入每轮候选轮次" }]}>
                                    <InputNumber min={1} max={10} style={{ width: "100%" }} />
                                  </Form.Item>
                                </Col>
                              </Row>
                              <Form.Item label="每轮自动补充新增因子数" name="automation_additional_factor_count_per_round" rules={[{ required: true, message: "请输入每轮新增因子数" }]}>
                                <InputNumber min={0} max={10} style={{ width: "100%" }} />
                              </Form.Item>
                              <Form.Item label="每轮因子更新方式" name="automation_factor_update_mode">
                                <Select>
                                  <Option value="append">追加新因子</Option>
                                  <Option value="reselect">根据上一轮结果重新选择因子</Option>
                                </Select>
                              </Form.Item>
                              <Card
                                size="small"
                                title="策略配置"
                                style={{ marginBottom: 16, background: "#fafafa", borderColor: "#f0f0f0" }}
                              >
                                <Form.Item
                                  label="下一轮父代选择策略"
                                  name="automation_parent_selection_strategy"
                                  tooltip="控制当本轮效果变差时，下一轮是沿用最新一轮继续，还是回到历史最高分结果继续。"
                                  style={{ marginBottom: 0 }}
                                >
                                  <Select>
                                    <Option value="best_score_so_far">如果本轮更差，则沿用历史最高分继续</Option>
                                    <Option value="latest_round">无论分数如何，始终沿用上一轮继续</Option>
                                  </Select>
                                </Form.Item>
                              </Card>
                              <Form.Item label="筛选逻辑" name="automation_match_mode">
                                <Select>
                                  <Option value="all">同时满足全部条件</Option>
                                  <Option value="any">满足任意一个条件</Option>
                                </Select>
                              </Form.Item>
                              <Row gutter={16}>
                                <Col xs={24} sm={12}>
                                  <Form.Item label="最低 Score" name="automation_score_min">
                                    <InputNumber min={0} max={100} style={{ width: "100%" }} />
                                  </Form.Item>
                                </Col>
                                <Col xs={24} sm={12}>
                                  <Form.Item label="最低 L/S Sharpe" name="automation_ls_sharpe_min">
                                    <InputNumber min={-5} max={10} step={0.1} style={{ width: "100%" }} />
                                  </Form.Item>
                                </Col>
                              </Row>
                              <Row gutter={16}>
                                <Col xs={24} sm={12}>
                                  <Form.Item label="最低 L/S Return" name="automation_ls_return_min">
                                    <InputNumber min={-1} max={5} step={0.01} style={{ width: "100%" }} />
                                  </Form.Item>
                                </Col>
                                <Col xs={24} sm={12}>
                                  <Form.Item label="最低 WQ Return" name="automation_wq_return_min">
                                    <InputNumber min={-1} max={5} step={0.01} style={{ width: "100%" }} />
                                  </Form.Item>
                                </Col>
                              </Row>
                              <Form.Item label="允许的 WQ Rating" name="automation_wq_ratings" tooltip="按 WQ Rating 筛选：Spectacular / Excellent / Good / Average / Needs Improvement">
                                <Select
                                  mode="multiple"
                                  placeholder="选择 WQ Rating；为空表示不限制"
                                  options={[
                                    { label: "Spectacular", value: "Spectacular" },
                                    { label: "Excellent", value: "Excellent" },
                                    { label: "Good", value: "Good" },
                                    { label: "Average", value: "Average" },
                                    { label: "Needs Improvement", value: "Needs Improvement" },
                                  ]}
                                />
                              </Form.Item>
                            </Card>
                          ) : null}
                        </>
                      ) : (
                        <>
                          <Card size="small" title="LLM 配置" loading={llmConfigLoading} className="llm-config-card">
                            <Form form={llmForm} layout="vertical">
                              <Form.Item label="API Key" name="api_key">
                                <Input.Password placeholder={llmConfig?.has_api_key ? `已配置：${llmConfig?.api_key_masked || ''}；留空则保持不变` : '填写 DeepSeek / OpenAI 兼容 API Key'} />
                              </Form.Item>
                              <Form.Item label="Base URL" name="base_url" rules={[{ required: true, message: '请输入 Base URL' }]}>
                                <Input placeholder="https://api.deepseek.com/v1" />
                              </Form.Item>
                              <Form.Item label="模型" name="model" rules={[{ required: true, message: '请输入模型名' }]}>
                                <Input placeholder="deepseek-chat" />
                              </Form.Item>
                              <Tag color={llmConfig?.has_api_key ? 'green' : 'orange'} style={{ marginBottom: 16 }}>
                                {llmConfig?.has_api_key ? 'LLM 已连接' : 'LLM 未配置'}
                              </Tag>
                              <div className="mobile-action-stack">
                                <Button type="primary" onClick={() => void saveLLMConfig()} loading={llmConfigSaving}>保存配置</Button>
                                <Button onClick={() => void restartLLMService()} loading={llmRestarting}>检查服务状态</Button>
                                <Button onClick={() => void loadLLMConfig()}>刷新状态</Button>
                              </div>
                            </Form>
                          </Card>

                          <Divider styles={{ content: { margin: 0 } }} titlePlacement="left">研究目标</Divider>
                          <Form.Item label="自动挖掘提示词" name="prompt" rules={[{ required: true, message: "请输入自动挖掘目标" }]}>
                            <Input.TextArea rows={4} placeholder="例如：从导入因子中自动挖掘更稳定的量价复合因子，优先提升 Sharpe 与稳定性" />
                          </Form.Item>
                          <Form.Item label="优化方向（可选）" name="direction">
                            <Select placeholder="选择本轮唯一主优化指标" options={OPTIMIZATION_DIRECTION_OPTIONS} allowClear />
                          </Form.Item>
                          <Form.Item label="最多因子数" name="max_factor_count" tooltip="LLM 会在这个上限内自主决定选择多少个基础因子，避免后续优化过于复杂">
                            <InputNumber min={1} max={50} style={{ width: "100%" }} />
                          </Form.Item>
                          <Form.Item>
                            <Button onClick={() => void handleLLMSelectFactors()} loading={llmSelectingFactors}>LLM 自动从因子库筛选因子</Button>
                          </Form.Item>
                          {llmSelectionSummary ? <Alert type="info" showIcon style={{ marginBottom: 16 }} message="LLM 已完成因子筛选" description={llmSelectionSummary} /> : null}

                          <Divider styles={{ content: { margin: 0 } }} titlePlacement="left">从因子库导入</Divider>
                          <Form.Item name="base_factors" rules={[{ required: true, message: "请至少导入一个基础因子" }]}>
                            <Select mode="multiple" placeholder="选择要导入的因子" showSearch optionLabelProp="label" maxTagCount="responsive">{renderFactorOptions()}</Select>
                          </Form.Item>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                            <span className="text-hint">已导入 <strong style={{ color: "#3b82f6" }}>{selectedFactorCount}</strong> 个因子</span>
                            <Space size="small">
                              <Button type="link" size="small" onClick={() => updateAutoBaseFactors(factors.map((f) => f.name))}>全选</Button>
                              <Button type="link" size="small" onClick={() => updateAutoBaseFactors([])}>清空</Button>
                            </Space>
                          </div>

                          <Divider styles={{ content: { margin: 0 } }} titlePlacement="left">研究参数</Divider>
                          <Form.Item label="日期范围" name="dateRange" rules={[{ required: true, message: "请选择日期范围" }]}>
                            <RangePicker
                              style={{ width: "100%" }}
                              allowClear
                              inputReadOnly={isMobileView}
                              placement={isMobileView ? "bottomLeft" : undefined}
                              getPopupContainer={(trigger) => trigger.parentElement || document.body}
                            />
                          </Form.Item>
                          <Row gutter={16}>
                            <Col xs={24} sm={12}><Form.Item label="股票池" name="universe"><Select><Option value="hs300">HS300</Option><Option value="zz500">ZZ500</Option><Option value="zz1000">ZZ1000</Option><Option value="all">全市场</Option></Select></Form.Item></Col>
                            <Col xs={24} sm={12}><Form.Item label="基准" name="benchmark"><Select><Option value="hs300">HS300</Option><Option value="zz500">ZZ500</Option><Option value="zz1000">ZZ1000</Option></Select></Form.Item></Col>
                          </Row>
                          <Row gutter={16}>
                            <Col xs={24} sm={12}><Form.Item label="分组数" name="n_groups"><InputNumber min={2} max={20} style={{ width: "100%" }} /></Form.Item></Col>
                            <Col xs={24} sm={12}><Form.Item label="持有期" name="holding_period"><InputNumber min={1} max={60} style={{ width: "100%" }} /></Form.Item></Col>
                          </Row>
                          <Form.Item label="候选轮次" name="n_candidates" tooltip="对应 FactorHub 本地候选表达式筛选轮次"><InputNumber min={1} max={10} style={{ width: "100%" }} /></Form.Item>
                          <Row gutter={16}>
                            <Col xs={24} sm={12}>
                              <Form.Item name="neutralize_industry" valuePropName="checked" style={{ marginBottom: 12 }}>
                                <Checkbox>行业中性化</Checkbox>
                              </Form.Item>
                            </Col>
                            <Col xs={24} sm={12}>
                              <Form.Item name="neutralize_cap" valuePropName="checked" style={{ marginBottom: 12 }}>
                                <Checkbox>市值中性化</Checkbox>
                              </Form.Item>
                            </Col>
                          </Row>

                          {autoLaunchMode === "campaign" ? (
                            <Card size="small" title="连续探索配置" style={{ marginBottom: 16 }}>
                              <Row gutter={16}>
                                <Col xs={24} sm={12}>
                                  <Form.Item label="累计探索次数" name="automation_exploration_rounds" rules={[{ required: true, message: "请输入累计探索次数" }]}>
                                    <InputNumber min={1} max={10} style={{ width: "100%" }} />
                                  </Form.Item>
                                </Col>
                                <Col xs={24} sm={12}>
                                  <Form.Item label="每轮候选轮次" name="automation_n_candidates_per_round" rules={[{ required: true, message: "请输入每轮候选轮次" }]}>
                                    <InputNumber min={1} max={10} style={{ width: "100%" }} />
                                  </Form.Item>
                                </Col>
                              </Row>
                              <Form.Item label="每轮自动补充新增因子数" name="automation_additional_factor_count_per_round" rules={[{ required: true, message: "请输入每轮新增因子数" }]}>
                                <InputNumber min={0} max={10} style={{ width: "100%" }} />
                              </Form.Item>
                              <Form.Item label="每轮因子更新方式" name="automation_factor_update_mode">
                                <Select>
                                  <Option value="append">追加新因子</Option>
                                  <Option value="reselect">根据上一轮结果重新选择因子</Option>
                                </Select>
                              </Form.Item>
                              <Card
                                size="small"
                                title="策略配置"
                                style={{ marginBottom: 16, background: "#fafafa", borderColor: "#f0f0f0" }}
                              >
                                <Form.Item
                                  label="下一轮父代选择策略"
                                  name="automation_parent_selection_strategy"
                                  tooltip="控制当本轮效果变差时，下一轮是沿用最新一轮继续，还是回到历史最高分结果继续。"
                                  style={{ marginBottom: 0 }}
                                >
                                  <Select>
                                    <Option value="best_score_so_far">如果本轮更差，则沿用历史最高分继续</Option>
                                    <Option value="latest_round">无论分数如何，始终沿用上一轮继续</Option>
                                  </Select>
                                </Form.Item>
                              </Card>
                              <Form.Item label="筛选逻辑" name="automation_match_mode">
                                <Select>
                                  <Option value="all">同时满足全部条件</Option>
                                  <Option value="any">满足任意一个条件</Option>
                                </Select>
                              </Form.Item>
                              <Row gutter={16}>
                                <Col xs={24} sm={12}>
                                  <Form.Item label="最低 Score" name="automation_score_min">
                                    <InputNumber min={0} max={100} style={{ width: "100%" }} />
                                  </Form.Item>
                                </Col>
                                <Col xs={24} sm={12}>
                                  <Form.Item label="最低 L/S Sharpe" name="automation_ls_sharpe_min">
                                    <InputNumber min={-5} max={10} step={0.1} style={{ width: "100%" }} />
                                  </Form.Item>
                                </Col>
                              </Row>
                              <Row gutter={16}>
                                <Col xs={24} sm={12}>
                                  <Form.Item label="最低 L/S Return" name="automation_ls_return_min">
                                    <InputNumber min={-1} max={5} step={0.01} style={{ width: "100%" }} />
                                  </Form.Item>
                                </Col>
                                <Col xs={24} sm={12}>
                                  <Form.Item label="最低 WQ Return" name="automation_wq_return_min">
                                    <InputNumber min={-1} max={5} step={0.01} style={{ width: "100%" }} />
                                  </Form.Item>
                                </Col>
                              </Row>
                              <Form.Item label="允许的 WQ Rating" name="automation_wq_ratings" tooltip="按 WQ Rating 筛选：Spectacular / Excellent / Good / Average / Needs Improvement">
                                <Select
                                  mode="multiple"
                                  placeholder="选择 WQ Rating；为空表示不限制"
                                  options={[
                                    { label: "Spectacular", value: "Spectacular" },
                                    { label: "Excellent", value: "Excellent" },
                                    { label: "Good", value: "Good" },
                                    { label: "Average", value: "Average" },
                                    { label: "Needs Improvement", value: "Needs Improvement" },
                                  ]}
                                />
                              </Form.Item>
                            </Card>
                          ) : null}

                        </>
                      )}

                      <Form.Item>
                        <Button
                          type="primary"
                          htmlType="submit"
                          icon={autoLaunchMode === "campaign" ? <RocketOutlined /> : <PlayCircleOutlined />}
                          loading={loading && activeTab === "auto"}
                          block
                          size="large"
                          disabled={mining}
                        >
                          {getAutoSubmitButtonLabel()}
                        </Button>
                      </Form.Item>
                    </Form>
                  </Card>
                </Col>
                <Col xs={24} lg={16}><Card title="自动挖掘结果" className="result-card">{renderStatusCard()}</Card></Col>
              </Row>
            ),
          },
          {
            key: "rdagent",
            label: "RDAgent 挖掘",
            children: renderRDAgentMiningTab(),
          },
        ]} />
      </div>
    </div>
  );
};

export default FactorMining;
