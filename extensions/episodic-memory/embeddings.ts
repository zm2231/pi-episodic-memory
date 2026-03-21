/**
 * Embeddings — local (Transformers.js) or remote (OpenAI-compatible HTTP endpoint).
 *
 * Remote mode is preferred when EPISODIC_EMBED_URL is set.
 * Falls back to local all-MiniLM-L6-v2 (384d) otherwise.
 *
 * Environment variables:
 *   EPISODIC_EMBED_URL    Base URL of OpenAI-compatible embeddings server
 *                         e.g. http://100.122.112.83:8100
 *   EPISODIC_EMBED_MODEL  Model name to request (default: bge-large-en-v1.5)
 *   EPISODIC_EMBED_DIM    Expected output dimension (default: auto-detected)
 */

// ─── Local fallback config ────────────────────────────────────────────────────
const LOCAL_MODEL_NAME = "Xenova/all-MiniLM-L6-v2";
const LOCAL_EMBEDDING_DIM = 384;

// ─── Remote config (from env) ─────────────────────────────────────────────────
const REMOTE_URL   = process.env.EPISODIC_EMBED_URL?.replace(/\/$/, "") ?? null;
const REMOTE_MODEL = process.env.EPISODIC_EMBED_MODEL ?? "bge-large-en-v1.5";

// EPISODIC_EMBED_DIM is required when using remote mode.
// We cannot probe the backend before DB schema creation, so the dim must be explicit.
const _rawDim = process.env.EPISODIC_EMBED_DIM;
let REMOTE_DIM: number | null = null;
if (REMOTE_URL) {
	if (!_rawDim) {
		throw new Error(
			"EPISODIC_EMBED_URL is set but EPISODIC_EMBED_DIM is missing. " +
			"Set EPISODIC_EMBED_DIM to the output dimension of your embedding model (e.g. 1024 for bge-large-en-v1.5).",
		);
	}
	if (!/^\d+$/.test(_rawDim.trim())) {
		throw new Error(`EPISODIC_EMBED_DIM must be a positive integer, got: ${JSON.stringify(_rawDim)}`);
	}
	const parsed = parseInt(_rawDim, 10);
	if (!Number.isFinite(parsed) || parsed <= 0) {
		throw new Error(`EPISODIC_EMBED_DIM must be a positive integer, got: ${JSON.stringify(_rawDim)}`);
	}
	REMOTE_DIM = parsed;
}

/** Resolved embedding dimension — known at startup */
export function getEmbeddingDim(): number {
	return REMOTE_DIM ?? LOCAL_EMBEDDING_DIM;
}

// ─── Remote embedding ─────────────────────────────────────────────────────────

async function embedRemote(texts: string[]): Promise<Float32Array[]> {
	const url = `${REMOTE_URL}/v1/embeddings`;
	const body = JSON.stringify({ model: REMOTE_MODEL, input: texts });

	const controller = new AbortController();
	const timeout = setTimeout(() => controller.abort(), 30_000);
	// Keep timeout active through body reads — server could stall mid-body
	let json: { data: { embedding: number[]; index: number }[] };
	try {
		const res = await fetch(url, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body,
			signal: controller.signal,
		});
		if (!res.ok) {
			throw new Error(`Remote embedding failed: ${res.status} ${await res.text()}`);
		}
		json = (await res.json()) as { data: { embedding: number[]; index: number }[] };
	} finally {
		clearTimeout(timeout);
	}

	if (!Array.isArray(json.data) || json.data.length !== texts.length) {
		throw new Error(
			`Remote embedding response has ${json.data?.length ?? "?"} items but ${texts.length} were requested.`,
		);
	}

	// Sort by index to ensure correct order and validate completeness
	const sorted = json.data.sort((a, b) => a.index - b.index);
	const expectedDim = REMOTE_DIM!;
	const results = sorted.map((d, i) => {
		if (d.index !== i) {
			throw new Error(
				`Remote embedding response has missing or duplicate index: expected ${i}, got ${d.index}.`,
			);
		}
		if (!Array.isArray(d.embedding) || d.embedding.length !== expectedDim) {
			throw new Error(
				`Remote embedding item ${i} has dim ${d.embedding?.length ?? "?"}, expected ${expectedDim}.`,
			);
		}
		return new Float32Array(d.embedding);
	});

	return results;
}

// ─── Local embedding ──────────────────────────────────────────────────────────

let localPipeline: any = null;
let localLoadingPromise: Promise<any> | null = null;

async function getLocalEmbedder(): Promise<any> {
	if (localPipeline) return localPipeline;
	if (localLoadingPromise) return localLoadingPromise;

	localLoadingPromise = (async () => {
		let createPipeline: any;
		try {
			const mod = await import("@huggingface/transformers");
			createPipeline = mod.pipeline;
		} catch {
			throw new Error(
				"@huggingface/transformers is not installed. " +
				"Either install it (`npm install @huggingface/transformers`) for local embedding mode, " +
				"or set EPISODIC_EMBED_URL to use a remote embedding server instead.",
			);
		}
		localPipeline = await createPipeline("feature-extraction", LOCAL_MODEL_NAME, { dtype: "fp32" });
		return localPipeline;
	})();

	return localLoadingPromise;
}

async function embedLocal(texts: string[]): Promise<Float32Array[]> {
	const embedder = await getLocalEmbedder();
	const results: Float32Array[] = [];
	for (const text of texts) {
		// Texts are already truncated by callers — no truncation here
		const result = await embedder(text, { pooling: "mean", normalize: true });
		results.push(new Float32Array(result.data));
	}
	return results;
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Generate an embedding for a single text.
 */
export async function embed(text: string): Promise<Float32Array> {
	const truncated = text.length > 2000 ? text.slice(0, 2000) : text;
	const results = REMOTE_URL
		? await embedRemote([truncated])
		: await embedLocal([truncated]);
	return results[0];
}

/**
 * Generate embeddings for multiple texts in batch.
 * Remote mode sends requests in chunks of 64 to avoid request size limits.
 * Local mode processes sequentially.
 */
export async function embedBatch(texts: string[]): Promise<Float32Array[]> {
	if (REMOTE_URL) {
		// Batch in chunks of 64 to avoid request size limits
		const CHUNK = 64;
		const results: Float32Array[] = [];
		for (let i = 0; i < texts.length; i += CHUNK) {
			const chunk = texts.slice(i, i + CHUNK).map((t) => (t.length > 2000 ? t.slice(0, 2000) : t));
			const batch = await embedRemote(chunk);
			results.push(...batch);
		}
		return results;
	}
	return embedLocal(texts);
}

/**
 * Serialize a Float32Array to a Buffer for SQLite storage.
 */
export function serializeEmbedding(embedding: Float32Array): Buffer {
	return Buffer.from(embedding.buffer, embedding.byteOffset, embedding.byteLength);
}

/**
 * Deserialize a Buffer from SQLite back to Float32Array.
 */
export function deserializeEmbedding(buffer: Buffer): Float32Array {
	const ab = new ArrayBuffer(buffer.length);
	const view = new Uint8Array(ab);
	for (let i = 0; i < buffer.length; i++) view[i] = buffer[i];
	return new Float32Array(ab);
}

export { REMOTE_URL, REMOTE_MODEL };
