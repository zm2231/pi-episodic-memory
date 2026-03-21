/**
 * Indexer: orchestrates parsing, embedding, and storing conversation chunks.
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import type { EpisodicMemoryDB } from "./database.js";
import { embed, serializeEmbedding } from "./embeddings.js";
import { chunkMessages, discoverSessionFiles, parseSessionFile, shouldExclude } from "./parser.js";

export interface IndexProgress {
	total: number;
	current: number;
	currentFile: string;
	newChunks: number;
}

// Sessions are always stored in the global ~/.pi/agent/sessions/ regardless of
// PI_CODING_AGENT_DIR — all pi instances share the same session store.
const SESSIONS_DIR = path.join(os.homedir(), ".pi", "agent", "sessions");

// DB is scoped to the active agent dir so each isolated pi environment keeps its
// own index.
// - Custom env (PI_CODING_AGENT_DIR set): <agentDir>/.pi/episodic-memory/index.db
//   (.pi/ is gitignored in repo-based agent dirs like pi-ult)
// - Default (~/.pi/agent): ~/.pi/agent/episodic-memory/index.db
//   (preserves the original location for standard pi installs)
const DB_DIR = process.env.PI_CODING_AGENT_DIR
	? path.join(process.env.PI_CODING_AGENT_DIR, ".pi", "episodic-memory")
	: path.join(os.homedir(), ".pi", "agent", "episodic-memory");
export const DB_PATH = path.join(DB_DIR, "index.db");

export { SESSIONS_DIR };

/**
 * Index all new/modified session files.
 * Returns the number of new chunks indexed.
 */
export async function indexNewSessions(
	db: EpisodicMemoryDB,
	onProgress?: (progress: IndexProgress) => void,
): Promise<{ filesIndexed: number; chunksIndexed: number }> {
	const allFiles = discoverSessionFiles(SESSIONS_DIR);
	let filesIndexed = 0;
	let chunksIndexed = 0;

	// Filter to only files that need indexing
	const filesToIndex: { path: string; stat: fs.Stats }[] = [];
	for (const filePath of allFiles) {
		try {
			const stat = fs.statSync(filePath);
			if (!db.isFileIndexed(filePath, stat.mtimeMs, stat.size)) {
				filesToIndex.push({ path: filePath, stat });
			}
		} catch {
			continue;
		}
	}

	if (filesToIndex.length === 0) return { filesIndexed: 0, chunksIndexed: 0 };

	for (let i = 0; i < filesToIndex.length; i++) {
		const file = filesToIndex[i];

		onProgress?.({
			total: filesToIndex.length,
			current: i + 1,
			currentFile: path.basename(file.path),
			newChunks: chunksIndexed,
		});

		// Check exclusion marker
		if (shouldExclude(file.path)) {
			db.markFileIndexed(file.path, file.stat.mtimeMs, file.stat.size);
			continue;
		}

		// Parse the session file
		const parsed = parseSessionFile(file.path);
		if (!parsed) {
			// null means read error — file may be mid-open or temporarily locked.
			// Don't mark as indexed; retry next session.
			console.error(`Failed to read session file, will retry: ${file.path}`);
			continue;
		}

		if (parsed.skippedLines > 0) {
			console.warn(`${file.path}: skipped ${parsed.skippedLines} malformed line(s), indexing partial session`);
		}

		if (!parsed.session) {
			// File was readable but had no session header — no way to attribute chunks.
			// Mark as indexed so we don't retry a permanently headerless file forever.
			if (parsed.skippedLines > 0) {
				console.warn(`${file.path}: no session header found, skipping`);
			}
			db.markFileIndexed(file.path, file.stat.mtimeMs, file.stat.size);
			continue;
		}

		if (parsed.messages.length === 0) {
			// Valid session record but no messages — mark indexed and skip
			db.markFileIndexed(file.path, file.stat.mtimeMs, file.stat.size);
			continue;
		}

		// Chunk the messages
		const chunks = chunkMessages(parsed.session, parsed.messages);

		// Phase 1: embed all chunks into memory first.
		// Do NOT touch the DB yet — if any embed fails, the existing index is preserved.
		type EmbeddedChunk = { chunk: (typeof chunks)[number]; embeddingBuf: Buffer };
		const embedded: EmbeddedChunk[] = [];
		let embedError: unknown = null;

		for (const chunk of chunks) {
			try {
				const embedding = await embed(chunk.text);
				embedded.push({ chunk, embeddingBuf: serializeEmbedding(embedding) });
			} catch (err) {
				embedError = err;
				console.error(`Failed to embed chunk ${chunk.chunkIndex} of ${file.path}:`, err);
				break; // abort remaining chunks — whole file will retry
			}
		}

		if (embedError) {
			// Embedding failed — leave existing DB data intact so previously indexed
			// content remains searchable. File will be retried on next session start.
			console.error(`Skipping DB write for ${file.path} — existing index preserved, will retry`);
			continue;
		}

		// Phase 2: all embeddings succeeded — atomically replace old data and insert new.
		db.transaction(() => {
			db.removeFile(file.path);
			for (const { chunk, embeddingBuf } of embedded) {
				db.insertChunk(
					chunk.session.id,
					chunk.session.filePath,
					chunk.session.project,
					chunk.session.cwd,
					chunk.chunkIndex,
					chunk.text,
					chunk.startTime,
					chunk.endTime,
					chunk.session.timestamp,
					embeddingBuf,
				);
			}
			db.markFileIndexed(file.path, file.stat.mtimeMs, file.stat.size);
		});

		chunksIndexed += embedded.length;
		filesIndexed++;
	}

	return { filesIndexed, chunksIndexed };
}
