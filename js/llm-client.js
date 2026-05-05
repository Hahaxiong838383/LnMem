const PROVIDER_CONFIG = {
  openai: {
    defaultModel: 'gpt-4o-mini',
    endpoint: 'https://api.openai.com/v1/chat/completions'
  },
  deepseek: {
    defaultModel: 'deepseek-chat',
    endpoint: 'https://api.deepseek.com/chat/completions'
  },
  anthropic: {
    apiStyle: 'anthropic-x-api-key',
    defaultModel: 'claude-3-5-sonnet-latest',
    endpoint: 'https://api.anthropic.com/v1/messages'
  },
  minimax: {
    apiStyle: 'anthropic-bearer',
    defaultModel: 'MiniMax-M2.7',
    endpoint: 'https://api.minimaxi.com/anthropic/v1/messages'
  }
};

function assertString(value, name) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`${name} must be a non-empty string`);
  }
}

function sanitizeApiKey(provider, apiKey) {
  let key = String(apiKey || '').trim();
  const normalizedProvider = String(provider || '').toLowerCase();

  // Users often paste keys with trailing punctuation from chat/apps.
  if (normalizedProvider === 'minimax' && /^sk-/i.test(key)) {
    key = key.replace(/[。．.,，;；:：!！?？"'`）)\]】》〉>]+$/u, '');
  }

  return key;
}

function getProviderConfig(provider) {
  const key = String(provider || '').toLowerCase();
  const config = PROVIDER_CONFIG[key];
  if (!config) {
    throw new Error(`Unsupported LLM provider: ${provider}. Expected openai, anthropic, deepseek, or minimax.`);
  }
  return { key, ...config };
}

function joinAnthropicText(content) {
  if (!Array.isArray(content)) return '';
  return content
    .filter((part) => part && part.type === 'text' && typeof part.text === 'string')
    .map((part) => part.text)
    .join('\n');
}

function normalizeOptions(options = {}) {
  return {
    temperature: options.temperature ?? 0.4,
    maxTokens: options.maxTokens ?? options.max_tokens ?? 4000,
    timeoutMs: options.timeoutMs ?? 90000,
    signal: options.signal,
    responseFormat: options.responseFormat,
    headers: options.headers || {},
    baseUrl: options.baseUrl,
    model: options.model
  };
}

function withTimeout(signal, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error('LLM request timed out')), timeoutMs);

  if (signal) {
    if (signal.aborted) controller.abort(signal.reason);
    signal.addEventListener('abort', () => controller.abort(signal.reason), { once: true });
  }

  return { signal: controller.signal, cancel: () => clearTimeout(timer) };
}

function withPromiseTimeout(promise, timeoutMs, message) {
  let timer;
  const timeoutPromise = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(message || 'LLM request timed out')), timeoutMs);
  });
  return Promise.race([promise, timeoutPromise]).finally(() => {
    clearTimeout(timer);
  });
}

function buildChatPayload(model, systemPrompt, userPrompt, options) {
  const payload = {
    model,
    temperature: options.temperature,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt }
    ]
  };

  if (options.maxTokens) payload.max_tokens = options.maxTokens;
  if (options.responseFormat) payload.response_format = options.responseFormat;

  return payload;
}

function buildAnthropicPayload(model, systemPrompt, userPrompt, options) {
  return {
    model,
    system: systemPrompt,
    temperature: options.temperature,
    max_tokens: options.maxTokens,
    messages: [{ role: 'user', content: userPrompt }]
  };
}

async function parseJsonResponse(response, providerKey) {
  const text = await response.text();
  let body;
  try {
    body = text ? JSON.parse(text) : {};
  } catch (error) {
    throw new Error(`${providerKey} returned non-JSON response: ${text.slice(0, 240)}`);
  }

  if (!response.ok) {
    const message = body?.error?.message || body?.message || response.statusText || 'LLM request failed';
    throw new Error(`${providerKey} API error ${response.status}: ${message}`);
  }

  return body;
}

/**
 * Call an LLM provider from a browser ESM environment.
 * Returns a normalized object: { provider, model, text, raw, usage }.
 */
export async function callLLM(provider, apiKey, systemPrompt, userPrompt, options = {}) {
  const config = getProviderConfig(provider);
  const normalized = normalizeOptions(options);
  const model = normalized.model || config.defaultModel;
  const endpoint = normalized.baseUrl || config.endpoint;
  const cleanApiKey = sanitizeApiKey(config.key, apiKey);

  assertString(cleanApiKey, 'apiKey');
  assertString(systemPrompt, 'systemPrompt');
  assertString(userPrompt, 'userPrompt');

  const { signal, cancel } = withTimeout(normalized.signal, normalized.timeoutMs);
  const commonHeaders = {
    'content-type': 'application/json',
    ...normalized.headers
  };

  try {
    let request;
    if (config.apiStyle === 'anthropic-x-api-key') {
      request = fetch(endpoint, {
        method: 'POST',
        signal,
        headers: {
          ...commonHeaders,
          'x-api-key': cleanApiKey,
          'anthropic-dangerous-direct-browser-access': normalized.headers['anthropic-dangerous-direct-browser-access'] || 'true',
          'anthropic-version': normalized.headers['anthropic-version'] || '2023-06-01'
        },
        body: JSON.stringify(buildAnthropicPayload(model, systemPrompt, userPrompt, normalized))
      });
    } else if (config.apiStyle === 'anthropic-bearer') {
      request = fetch(endpoint, {
        method: 'POST',
        signal,
        headers: {
          ...commonHeaders,
          authorization: `Bearer ${cleanApiKey}`
        },
        body: JSON.stringify(buildAnthropicPayload(model, systemPrompt, userPrompt, normalized))
      });
    } else {
      request = fetch(endpoint, {
        method: 'POST',
        signal,
        headers: {
          ...commonHeaders,
          authorization: `Bearer ${cleanApiKey}`
        },
        body: JSON.stringify(buildChatPayload(model, systemPrompt, userPrompt, normalized))
      });
    }

    const response = await withPromiseTimeout(
      request,
      normalized.timeoutMs + 5000,
      'LLM request timed out'
    );
    const raw = await parseJsonResponse(response, config.key);
    const text = config.apiStyle?.startsWith('anthropic')
      ? joinAnthropicText(raw.content)
      : (raw.choices?.[0]?.message?.content || raw.choices?.[0]?.text || '');

    if (!text) {
      throw new Error(`${config.key} API response did not contain text content`);
    }

    return {
      provider: config.key,
      model,
      text,
      raw,
      usage: raw.usage || null
    };
  } finally {
    cancel();
  }
}

export default callLLM;
