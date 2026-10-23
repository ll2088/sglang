#!/usr/bin/env node
/**
 * autojump validation runner.
 *
 * Installed into the target repository and executed once per matrix dimension
 * (effect / performance / stability / usability). Zero external dependencies.
 */
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, appendFileSync, mkdtempSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const DIMENSIONS = {
  effect: { label: '效果', runs: 1 },
  performance: { label: '性能', runs: 3 },
  stability: { label: '稳定性', runs: 3 },
  usability: { label: '可用性', runs: 1 },
};
// The workflow checks the runner out separately; fall back to the repo root for local runs.
const CONFIG_PATH =
  process.env.MERGELENS_CONFIG && existsSync(process.env.MERGELENS_CONFIG)
    ? process.env.MERGELENS_CONFIG
    : '.self-improvement.json';
const NODE_TOOL = /^(?:\S+=\S+\s+)*(?:npm|npx|pnpm|yarn|bun)\b/;
const dimension = String(process.argv[2] ?? '').trim().toLowerCase();

function fail(message) {
  console.error(`\n[autojump] ERROR: ${message}\n`);
  if (process.env.GITHUB_ACTIONS === 'true') {
    console.error(`::error title=autojump validation::${escapeAnnotation(message)}`);
  }
  process.exitCode = 1;
}

function escapeAnnotation(value) {
  return String(value).replace(/%/g, '%25').replace(/\r/g, '%0D').replace(/\n/g, '%0A');
}

function readConfig() {
  if (!existsSync(CONFIG_PATH)) {
    throw new Error(`未找到 ${CONFIG_PATH}。请从 autojump 设置页安装验证资产。`);
  }
  const raw = readFileSync(CONFIG_PATH, 'utf8');
  let config;
  try {
    config = JSON.parse(raw);
  } catch (error) {
    throw new Error(`${CONFIG_PATH} 不是合法 JSON：${error.message}`);
  }
  if (!config || typeof config !== 'object') throw new Error(`${CONFIG_PATH} 根节点必须是对象`);
  return config;
}

function normalizeDimensionConfig(config, key) {
  const raw = config[key];
  if (typeof raw === 'string') return { command: raw };
  if (raw && typeof raw === 'object') return raw;
  return {};
}

function autoSetupCommand(cwd) {
  if (!existsSync(path.join(cwd, 'package.json'))) return '';
  if (existsSync(path.join(cwd, 'package-lock.json'))) return 'npm ci';
  if (existsSync(path.join(cwd, 'pnpm-lock.yaml'))) {
    return 'corepack enable && pnpm install --frozen-lockfile';
  }
  if (existsSync(path.join(cwd, 'yarn.lock'))) {
    return 'corepack enable && yarn install --immutable';
  }
  if (existsSync(path.join(cwd, 'bun.lockb')) || existsSync(path.join(cwd, 'bun.lock'))) {
    return 'bun install --frozen-lockfile';
  }
  return 'npm install';
}

// Output still streams live; a copy is kept so a failure can quote its last lines.
const LOG_DIR = mkdtempSync(path.join(os.tmpdir(), 'mergelens-'));
process.on('exit', () => rmSync(LOG_DIR, { recursive: true, force: true }));
const TEE_SCRIPT = 'set -o pipefail; /bin/bash -lc "$MERGELENS_CMD" 2>&1 | tee "$MERGELENS_LOG"';

function outputTail(file, lines = 25) {
  try {
    return readFileSync(file, 'utf8').replace(/\x1b\[[0-9;]*m/g, '').trimEnd().split('\n').slice(-lines).join('\n');
  } catch {
    return '';
  }
}

function runShell(command, { timeoutMs, label, env = {} }) {
  const started = process.hrtime.bigint();
  console.log(`\n$ ${command}`);
  const logFile = path.join(LOG_DIR, `${label.replace(/[^\w.-]+/g, '_')}.log`);
  const result = spawnSync('/bin/bash', ['-c', TEE_SCRIPT], {
    cwd: process.cwd(),
    stdio: 'inherit',
    env: {
      ...process.env,
      CI: '1',
      MERGELENS_DIMENSION: dimension,
      ...env,
      MERGELENS_CMD: command,
      MERGELENS_LOG: logFile,
    },
    timeout: timeoutMs,
    killSignal: 'SIGTERM',
  });
  const durationMs = Number(process.hrtime.bigint() - started) / 1e6;
  const timedOut = result.error?.code === 'ETIMEDOUT' || result.signal === 'SIGTERM';
  const ok = result.status === 0 && !result.error;
  console.log(
    `[autojump] ${label} ${ok ? 'PASS' : 'FAIL'} · exit=${result.status ?? 'n/a'} · ${Math.round(durationMs)}ms${
      timedOut ? ' · timeout' : ''
    }`,
  );
  return {
    ok,
    status: result.status,
    signal: result.signal,
    timedOut,
    durationMs,
    error: result.error ? result.error.message : null,
    tail: ok ? '' : outputTail(logFile),
  };
}

/** Puts the failing command's last output lines in the run's Annotations panel. */
function annotateFailure(title, result) {
  const reason = result.timedOut ? '超时' : `exit=${result.status ?? result.signal ?? 'n/a'}`;
  fail(`${title}失败（${reason}）${result.tail ? `，输出末尾：\n${result.tail}` : ''}`);
}

function median(values) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function appendSummary(lines) {
  const markdown = `${lines.join('\n')}\n`;
  console.log(`\n${markdown}`);
  if (process.env.GITHUB_STEP_SUMMARY) {
    try {
      appendFileSync(process.env.GITHUB_STEP_SUMMARY, markdown, 'utf8');
    } catch (error) {
      console.warn(`[autojump] 无法写入 job summary: ${error.message}`);
    }
  }
}

function main() {
  const meta = DIMENSIONS[dimension];
  if (!meta) {
    console.error(`Usage: node run-dimension.mjs <${Object.keys(DIMENSIONS).join('|')}>`);
    process.exitCode = 2;
    return;
  }
  console.log(`[autojump] dimension=${dimension} cwd=${process.cwd()}`);
  console.log(
    `[autojump] validation=${process.env.MERGELENS_VALIDATION_ID ?? 'local'} candidate=${
      process.env.MERGELENS_CANDIDATE_ID ?? 'local'
    }`,
  );

  let config;
  try {
    config = readConfig();
  } catch (error) {
    fail(error.message);
    appendSummary([`## autojump · ${meta.label}`, '', `❌ ${error.message}`]);
    return;
  }

  const dimensionConfig = normalizeDimensionConfig(config, dimension);
  const command = String(dimensionConfig.command ?? '').trim();
  if (!command) {
    const message = `「${meta.label}」尚未配置命令。请在 autojump 设置页填写 ${dimension} 命令并重新安装验证资产，或直接修改 ${CONFIG_PATH}。`;
    fail(message);
    appendSummary([`## autojump · ${meta.label}`, '', `⚠️ ${message}`]);
    return;
  }

  // The defaults are Node commands; on a Python / Go / Rust repo they can only fail.
  if (NODE_TOOL.test(command) && !existsSync('package.json')) {
    const message = `仓库根目录没有 package.json，但「${meta.label}」命令是 \`${command}\`。请在 autojump 项目「流水线」页把命令改成适合本项目的写法（例如 Python 项目用 \`pytest\`），保存后点击「安装 / 更新」。`;
    fail(message);
    appendSummary([`## autojump · ${meta.label}`, '', `⚠️ ${message}`]);
    return;
  }

  const timeoutMinutes = Number(dimensionConfig.timeout_minutes) || 20;
  const timeoutMs = Math.max(1, timeoutMinutes) * 60_000;
  const runsRequested = Math.max(
    1,
    Math.min(10, Number(dimensionConfig.runs) || meta.runs),
  );

  const setup = String(config.setup ?? '').trim() || autoSetupCommand(process.cwd());
  const summary = [
    `## autojump · ${meta.label} (${dimension})`,
    '',
    `- 候选：\`${process.env.MERGELENS_CANDIDATE_ID ?? 'local'}\``,
    `- 验证 ID：\`${process.env.MERGELENS_VALIDATION_ID ?? 'local'}\``,
    `- 维度：**${meta.label}** · 命令执行 ${runsRequested} 次`,
    '',
  ];

  if (setup) {
    console.log(`[autojump] setup: ${setup}`);
    const setupResult = runShell(setup, {
      timeoutMs: Math.min(timeoutMs, 30 * 60_000),
      label: 'setup',
    });
    summary.push(
      `### 环境准备`,
      '',
      setupResult.ok ? '✅ 成功' : '❌ 失败',
      '',
      `<details><summary>setup command</summary>`,
      '',
      '```bash',
      setup,
      '```',
      '',
      '</details>',
      '',
    );
    if (!setupResult.ok) {
      annotateFailure('环境准备命令', setupResult);
      summary.push('```text', setupResult.tail, '```', '');
      appendSummary(summary);
      return;
    }
  }

  const results = [];
  if (dimension === 'stability') {
    for (let index = 1; index <= runsRequested; index += 1) {
      const result = runShell(command, {
        timeoutMs,
        label: `run ${index}/${runsRequested}`,
        env: { MERGELENS_RUN_INDEX: String(index) },
      });
      results.push(result);
    }
  } else {
    for (let index = 1; index <= runsRequested; index += 1) {
      const result = runShell(command, {
        timeoutMs,
        label: `run ${index}/${runsRequested}`,
        env: { MERGELENS_RUN_INDEX: String(index) },
      });
      results.push(result);
      if (!result.ok) break;
    }
  }

  const durations = results.map((result) => result.durationMs);
  const failures = results.filter((result) => !result.ok).length;
  const failureRate = results.length ? failures / results.length : 1;
  const maxFailureRate = Number.isFinite(Number(dimensionConfig.max_failure_rate))
    ? Number(dimensionConfig.max_failure_rate)
    : 0;

  summary.push(
    `### 执行结果`,
    '',
    '| # | 结果 | 耗时 |',
    '| --- | --- | --- |',
    ...results.map(
      (result, index) =>
        `| ${index + 1} | ${result.ok ? '✅ 通过' : `❌ 失败${result.timedOut ? '（超时）' : ''}`} | ${Math.round(
          result.durationMs,
        )}ms |`,
    ),
    '',
  );

  let passed = failures === 0;
  if (dimension === 'performance') {
    const medianMs = median(durations);
    const budgetMs = Number(dimensionConfig.budget_ms) || 0;
    // Machine-readable metric for autojump's baseline comparison (read back via check-run annotations).
    if (process.env.GITHUB_ACTIONS === 'true' && durations.length) {
      console.log(
        `::notice title=hop-metric::${JSON.stringify({
          dimension,
          medianMs: Math.round(medianMs),
          runs: durations.length,
          samples: durations.map((value) => Math.round(value)),
        })}`,
      );
    }
    summary.push(
      `- 中位耗时：**${Math.round(medianMs)}ms**`,
      `- 性能预算：${budgetMs > 0 ? `**${Math.round(budgetMs)}ms**` : '未设置'}`,
      '',
    );
    if (budgetMs > 0 && medianMs > budgetMs) {
      passed = false;
      summary.push(`❌ 中位耗时超出预算 ${Math.round(medianMs - budgetMs)}ms。`, '');
    }
  }

  if (dimension === 'stability') {
    summary.push(
      `- 失败率：**${(failureRate * 100).toFixed(1)}%**（上限 ${(maxFailureRate * 100).toFixed(1)}%）`,
      `- 运行次数：**${results.length}**`,
      '',
    );
    if (failureRate > maxFailureRate) {
      passed = false;
      summary.push('❌ 失败率超过配置上限。', '');
    }
  }

  summary.push(
    `<details><summary>执行命令</summary>`,
    '',
    '```bash',
    command,
    '```',
    '',
    '</details>',
    '',
    passed ? '### ✅ 验证通过' : '### ❌ 验证未通过',
  );

  const firstFailure = results.find((result) => !result.ok);
  if (firstFailure?.tail) summary.push('', '<details open><summary>首次失败的输出末尾</summary>', '', '```text', firstFailure.tail, '```', '', '</details>');
  appendSummary(summary);
  if (!passed) {
    console.error(`[autojump] ${dimension} validation failed (${failures}/${results.length} runs).`);
    if (firstFailure) annotateFailure(`「${meta.label}」命令`, firstFailure);
    process.exitCode = 1;
  }
}

main();
