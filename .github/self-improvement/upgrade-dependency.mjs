#!/usr/bin/env node
/**
 * autojump dependency upgrade step.
 *
 * Runs inside the project's GitHub Actions. Inputs arrive as environment
 * variables (never interpolated into a shell), every external command uses an
 * argument array, and the only branch it ever pushes is `mergelens/deps/*`.
 * Zero external dependencies.
 */
import { spawnSync } from 'node:child_process';
import { appendFileSync, existsSync, readFileSync, writeFileSync } from 'node:fs';

const input = {
  id: env('MERGELENS_UPGRADE_ID'),
  ecosystem: env('MERGELENS_ECOSYSTEM'),
  name: env('MERGELENS_PACKAGE'),
  version: env('MERGELENS_VERSION'),
  spec: env('MERGELENS_SPEC'),
  manifest: env('MERGELENS_MANIFEST'),
  branch: env('MERGELENS_BRANCH'),
};

function env(key) {
  return String(process.env[key] ?? '').trim();
}

function fail(message) {
  console.error(`::error title=autojump upgrade::${String(message).replace(/\r?\n/g, '%0A')}`);
  summary([`### ❌ autojump 升级失败`, '', message]);
  process.exit(1);
}

function summary(lines) {
  const text = `${lines.join('\n')}\n`;
  console.log(text);
  if (process.env.GITHUB_STEP_SUMMARY) appendFileSync(process.env.GITHUB_STEP_SUMMARY, text, 'utf8');
}

function run(command, args, { allowFailure = false } = {}) {
  console.log(`\n$ ${command} ${args.join(' ')}`);
  const result = spawnSync(command, args, { stdio: 'inherit', env: { ...process.env, CI: '1' } });
  if (result.error && result.error.code === 'ENOENT') {
    if (allowFailure) return false;
    fail(`找不到命令 ${command}`);
  }
  if (result.status !== 0) {
    if (allowFailure) return false;
    fail(`${command} ${args.join(' ')} 退出码 ${result.status}`);
  }
  return true;
}

function has(command) {
  return spawnSync(command, ['--version'], { stdio: 'ignore' }).status === 0;
}

function validateInputs() {
  if (!/^mergelens\/deps\/[a-z0-9][a-z0-9.-]*$/.test(input.branch)) fail(`拒绝推送非 autojump 分支：${input.branch}`);
  if (!['npm', 'go', 'pypi'].includes(input.ecosystem)) fail(`不支持的生态：${input.ecosystem}`);
  if (!/^[A-Za-z0-9@][A-Za-z0-9@/._~+-]{0,213}$/.test(input.name)) fail(`依赖名不合法：${input.name}`);
  if (!/^v?[0-9][0-9A-Za-z.+!-]{0,63}$/.test(input.version)) fail(`版本号不合法：${input.version}`);
  if (/[\s;&|`$<>"'\\]/.test(input.spec)) fail(`版本约束不合法：${input.spec}`);
  const manifests = ['package.json', 'go.mod', 'requirements.txt', 'pyproject.toml'];
  if (!manifests.includes(input.manifest)) fail(`不支持的清单文件：${input.manifest}`);
  if (!existsSync(input.manifest)) fail(`仓库根目录没有 ${input.manifest}`);
}

/* ---------------------------------- npm ---------------------------------- */

function upgradeNpm() {
  const pkg = JSON.parse(readFileSync('package.json', 'utf8'));
  let found = false;
  for (const scope of ['dependencies', 'devDependencies', 'optionalDependencies', 'peerDependencies']) {
    if (pkg[scope] && Object.prototype.hasOwnProperty.call(pkg[scope], input.name)) {
      pkg[scope][input.name] = input.spec;
      found = true;
    }
  }
  if (!found) fail(`package.json 中没有 ${input.name}`);
  const original = readFileSync('package.json', 'utf8');
  const indent = original.match(/^\{\n([ \t]+)"/)?.[1] ?? '  ';
  writeFileSync('package.json', `${JSON.stringify(pkg, null, indent)}${original.endsWith('\n') ? '\n' : ''}`);

  // Refresh only the lockfile that the repository already uses.
  if (existsSync('pnpm-lock.yaml')) {
    run('corepack', ['enable']);
    run('pnpm', ['install', '--lockfile-only', '--ignore-scripts']);
  } else if (existsSync('yarn.lock')) {
    run('corepack', ['enable']);
    if (!run('yarn', ['install', '--mode', 'update-lockfile'], { allowFailure: true })) {
      run('yarn', ['install', '--ignore-scripts']);
    }
  } else if (existsSync('bun.lockb') || existsSync('bun.lock')) {
    run('npm', ['install', '--global', 'bun']);
    run('bun', ['install', '--ignore-scripts']);
  } else if (existsSync('package-lock.json')) {
    run('npm', ['install', '--package-lock-only', '--ignore-scripts', '--no-audit', '--no-fund']);
  }
}

/* ----------------------------------- go ---------------------------------- */

function upgradeGo() {
  run('go', ['get', `${input.name}@${input.version}`]);
  run('go', ['mod', 'tidy']);
}

/* --------------------------------- python -------------------------------- */

function normalize(name) {
  return name.toLowerCase().replace(/[-_.]+/g, '-');
}

/** Replaces the version part of one requirement string, keeping extras and environment markers. */
function rewriteRequirement(text) {
  const match = text.match(/^(\s*)([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?([^;]*)(;.*)?$/s);
  if (!match || normalize(match[2]) !== normalize(input.name)) return null;
  const marker = match[5] ? ` ${match[5].trim()}` : '';
  return `${match[1]}${match[2]}${match[3] ?? ''}${input.spec}${marker}`;
}

function upgradePython() {
  const original = readFileSync(input.manifest, 'utf8');
  let changed = false;
  let next;
  if (input.manifest === 'requirements.txt') {
    next = original
      .split('\n')
      .map((line) => {
        const [body, ...comment] = line.split(/(?=\s+#)/);
        const rewritten = rewriteRequirement(body);
        if (rewritten === null) return line;
        changed = true;
        return `${rewritten}${comment.join('')}`;
      })
      .join('\n');
  } else {
    next = original.replace(/(["'])([^"'\n]+)\1/g, (whole, quote, value) => {
      const rewritten = rewriteRequirement(value);
      if (rewritten === null || !/[=<>~!]/.test(value)) return whole;
      changed = true;
      return `${quote}${rewritten}${quote}`;
    });
  }
  if (!changed) fail(`${input.manifest} 中没有可改写的 ${input.name} 版本约束`);
  writeFileSync(input.manifest, next);
  if (existsSync('uv.lock') && (has('uv') || run('pip', ['install', 'uv'], { allowFailure: true }))) {
    run('uv', ['lock', '--upgrade-package', input.name]);
  } else if (existsSync('poetry.lock') && (has('poetry') || run('pip', ['install', 'poetry'], { allowFailure: true }))) {
    run('poetry', ['lock', '--no-update']);
  }
}

/* ---------------------------------- main --------------------------------- */

function main() {
  validateInputs();
  console.log(`[autojump] upgrade=${input.id} ${input.ecosystem} ${input.name} → ${input.version}`);
  if (input.ecosystem === 'npm') upgradeNpm();
  if (input.ecosystem === 'go') upgradeGo();
  if (input.ecosystem === 'pypi') upgradePython();

  const status = spawnSync('git', ['status', '--porcelain'], { encoding: 'utf8' }).stdout.trim();
  if (!status) fail('升级后没有任何文件变化，可能已经是该版本');

  run('git', ['config', 'user.name', 'mergelens[bot]']);
  run('git', ['config', 'user.email', 'mergelens-bot@users.noreply.github.com']);
  run('git', ['checkout', '-B', input.branch]);
  run('git', ['add', '-A']);
  run('git', ['commit', '-m', `chore(deps): bump ${input.name} to ${input.version}`, '-m', `autojump upgrade ${input.id}`]);
  // The branch is owned by autojump; re-running an upgrade replaces it.
  run('git', ['push', '--force', 'origin', `HEAD:refs/heads/${input.branch}`]);

  summary([
    '### ✅ autojump 依赖升级完成',
    '',
    `- 依赖：\`${input.name}\` → \`${input.version}\`（${input.ecosystem}）`,
    `- 分支：\`${input.branch}\``,
    '- 下一步：autojump 会在该分支上触发四项验证，通过后提交 PR。',
  ]);
}

main();
