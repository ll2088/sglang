#!/usr/bin/env node
/**
 * MergeLens validation runner.
 *
 * Installed into the target repository and executed once per matrix dimension
 * (effect / performance / stability / usability). Zero external dependencies.
 */
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, appendFileSync } from 'node:fs';
import path from 'node:path';

const DIMENSIONS = {
  effect: { label: '效果', runs: 1 },
  performance: { label: '性能', runs: 3 },
  stability: { label: '稳定性', runs: 3 },
  usability: { label: '可用性', runs: 1 },
};
const CONFIG_PATH = '.self-improvement.json';
const dimension = String(process.argv[2] ?? '').trim().toLowerCase();

function fail(message) {
  console.error(`\n[MergeLens] ERROR: ${message}\n`);
  if (process.env.GITHUB_ACTIONS === 'true') {
    console.error(`::error title=MergeLens validation::${escapeAnnotation(message)}`);
  }
  process.exitCode = 1;
}

function escapeAnnotation(value) {
  return String(value).replace(/%/g, '%25').replace(/\r/g, '%0D').replace(/\n/g, '%0A');
}

function readConfig() {
  if (!existsSync(CONFIG_PATH)) {
    throw new Error(`未找到 ${CONFIG_PATH}。请从 MergeLens 设置页安装验证资产。`);
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

function runShell(command, { timeoutMs, label, env = {} }) {
  const started = process.hrtime.bigint();
  console.log(`\n$ ${command}`);
  const result = spawnSync('/bin/bash', ['-lc', command], {
    cwd: process.cwd(),
    stdio: 'inherit',
    env: {
      ...process.env,
      CI: '1',
      MERGELENS_DIMENSION: dimension,
      ...env,
    },
    timeout: timeoutMs,
    killSignal: 'SIGTERM',
  });
  const durationMs = Number(process.hrtime.bigint() - started) / 1e6;
  const timedOut = result.error?.code === 'ETIMEDOUT' || result.signal === 'SIGTERM';
  const ok = result.status === 0 && !result.error;
  console.log(
    `[MergeLens] ${label} ${ok ? 'PASS' : 'FAIL'} · exit=${result.status ?? 'n/a'} · ${Math.round(durationMs)}ms${
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
  };
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
      console.warn(`[MergeLens] 无法写入 job summary: ${error.message}`);
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
  console.log(`[MergeLens] dimension=${dimension} cwd=${process.cwd()}`);
  console.log(
    `[MergeLens] validation=${process.env.MERGELENS_VALIDATION_ID ?? 'local'} candidate=${
      process.env.MERGELENS_CANDIDATE_ID ?? 'local'
    }`,
  );

  let config;
  try {
    config = readConfig();
  } catch (error) {
    fail(error.message);
    appendSummary([`## MergeLens · ${meta.label}`, '', `❌ ${error.message}`]);
    return;
  }

  const dimensionConfig = normalizeDimensionConfig(config, dimension);
  const command = String(dimensionConfig.command ?? '').trim();
  if (!command) {
    const message = `「${meta.label}」尚未配置命令。请在 MergeLens 设置页填写 ${dimension} 命令并重新安装验证资产，或直接修改 ${CONFIG_PATH}。`;
    fail(message);
    appendSummary([`## MergeLens · ${meta.label}`, '', `⚠️ ${message}`]);
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
    `## MergeLens · ${meta.label} (${dimension})`,
    '',
    `- 候选：\`${process.env.MERGELENS_CANDIDATE_ID ?? 'local'}\``,
    `- 验证 ID：\`${process.env.MERGELENS_VALIDATION_ID ?? 'local'}\``,
    `- 维度：**${meta.label}** · 命令执行 ${runsRequested} 次`,
    '',
  ];

  if (setup) {
    console.log(`[MergeLens] setup: ${setup}`);
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
      fail(`环境准备命令失败（exit=${setupResult.status ?? 'timeout'}）`);
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

  appendSummary(summary);
  if (!passed) {
    console.error(`[MergeLens] ${dimension} validation failed (${failures}/${results.length} runs).`);
    process.exitCode = 1;
  }
}

main();
