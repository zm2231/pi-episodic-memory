/**
 * SQLite database with sqlite-vec for vector similarity search.
 * Stores conversation chunks and their embeddings.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import Database from "better-sqlite3";
import * as sqliteVec from "sqlite-vec";
import { getEmbeddingDim } from "./embeddings.js";

export interface StoredChunk {
	id: number;
	sessionId: string;
	sessionFile: string;
	project: string;
	cwd: string;
	chunkIndex: number;
	text: string;
	startTime: string;
	endTime: string;
	sessionTimestamp: string;
}

export interface SearchResult extends StoredChunk {
	distance: number;
	score: number; // 0-1, higher is better
}

/**
 * Normalize a project identifier for comparison.
 *
 * Sessions are stored with project = raw directory name, e.g. "--Volumes-4-GitHub-pi-ult--".
 * The encoding replaces "/" with "-", so path separators and hyphens in names are
 * indistinguishable. Full path decoding is therefore not reliable.
 *
 * We normalize by stripping leading/trailing "--", lowercasing, and treating the
 * result as an opaque string for suffix matching.
 */
function normalizeProjectName(name: string): string {
	return name
		.replace(/^-+/, "")
		.replace(/-+$/, "")
		.toLowerCase();
}

/**
 * Returns true if the stored raw project value matches the user-supplied filter.
 *
 * Supports short repo names only (e.g. "pi-ult").
 * Full decoded paths (e.g. "Volumes/4/GitHub/pi-ult") are NOT supported because
 * the encoding is lossy — "/" and "-" map to the same character.
 *
 * Examples:
 *   projectMatches("--Volumes-4-GitHub-pi-ult--", "pi-ult")  → true
 *   projectMatches("--Volumes-4-GitHub-pi-ult--", "pi-ult")  → true (exact suffix)
 *   projectMatches("--Users-zain-zira--", "pi-ult")          → false
 */
function projectMatches(storedRaw: string, filter: string): boolean {
	const stored = normalizeProjectName(storedRaw);
	const needle = normalizeProjectName(filter);
	return stored === needle || stored.endsWith("-" + needle);
}

export class EpisodicMemoryDB {
	private db: Database.Database;

	constructor(dbPath: string) {
		// Ensure directory exists
		const dir = path.dirname(dbPath);
		if (!fs.existsSync(dir)) {
			fs.mkdirSync(dir, { recursive: true });
		}

		this.db = new Database(dbPath);
		try {
			this.db.pragma("journal_mode = WAL");
			sqliteVec.load(this.db);
			this.init();
		} catch (err) {
			this.db.close();
			throw err;
		}
	}

	/**
	 * Read the embedding dimension from the existing vec0 table, or null if not yet created.
	 */
	private getExistingVecDim(): number | null {
		try {
			// sqlite-vec stores schema in sqlite_master; the CREATE statement has float[N]
			const row = this.db
				.prepare("SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_vec'")
				.get() as { sql: string } | undefined;
			if (!row) return null;
			const match = row.sql.match(/float\[(\d+)\]/);
			return match ? parseInt(match[1], 10) : null;
		} catch {
			return null;
		}
	}

	private init() {
		// Main chunks table
		this.db.exec(`
			CREATE TABLE IF NOT EXISTS chunks (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				session_id TEXT NOT NULL,
				session_file TEXT NOT NULL,
				project TEXT NOT NULL,
				cwd TEXT NOT NULL,
				chunk_index INTEGER NOT NULL,
				text TEXT NOT NULL,
				start_time TEXT NOT NULL,
				end_time TEXT NOT NULL,
				session_timestamp TEXT NOT NULL,
				UNIQUE(session_file, chunk_index)
			)
		`);

		// Track which files have been indexed
		this.db.exec(`
			CREATE TABLE IF NOT EXISTS indexed_files (
				file_path TEXT PRIMARY KEY,
				mtime_ms INTEGER NOT NULL,
				size INTEGER NOT NULL,
				indexed_at TEXT NOT NULL
			)
		`);

		// Virtual table for vector search — dimension is dynamic based on embedding backend.
		// If the table already exists with a different dimension, we detect and throw early
		// rather than silently inserting incompatible vectors.
		const dim = getEmbeddingDim();
		const existingDim = this.getExistingVecDim();
		if (existingDim !== null && existingDim !== dim) {
			throw new Error(
				`Embedding dimension mismatch: DB has ${existingDim}d but current backend produces ${dim}d. ` +
				`Run /memory-reindex to rebuild the index with the new dimensions.`,
			);
		}
		this.db.exec(`
			CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
				chunk_id INTEGER PRIMARY KEY,
				embedding float[${dim}]
			)
		`);

		// Indexes for filtering
		this.db.exec(`CREATE INDEX IF NOT EXISTS idx_chunks_text ON chunks(text)`);
		this.db.exec(`CREATE INDEX IF NOT EXISTS idx_chunks_project ON chunks(project)`);
		this.db.exec(`CREATE INDEX IF NOT EXISTS idx_chunks_session_timestamp ON chunks(session_timestamp)`);
	}

	/**
	 * Check if a file needs (re)indexing based on mtime and size.
	 */
	isFileIndexed(filePath: string, mtimeMs: number, size: number): boolean {
		const row = this.db.prepare("SELECT mtime_ms, size FROM indexed_files WHERE file_path = ?").get(filePath) as
			| { mtime_ms: number; size: number }
			| undefined;
		if (!row) return false;
		return row.mtime_ms === mtimeMs && row.size === size;
	}

	/**
	 * Remove all chunks for a given session file (for reindexing).
	 */
	removeFile(filePath: string) {
		const chunks = this.db.prepare("SELECT id FROM chunks WHERE session_file = ?").all(filePath) as { id: number }[];
		const deleteVec = this.db.prepare("DELETE FROM chunks_vec WHERE chunk_id = CAST(? AS INTEGER)");
		for (const chunk of chunks) {
			deleteVec.run(chunk.id);
		}
		this.db.prepare("DELETE FROM chunks WHERE session_file = ?").run(filePath);
		this.db.prepare("DELETE FROM indexed_files WHERE file_path = ?").run(filePath);
	}

	/**
	 * Insert a chunk and its embedding.
	 */
	insertChunk(
		sessionId: string,
		sessionFile: string,
		project: string,
		cwd: string,
		chunkIndex: number,
		text: string,
		startTime: string,
		endTime: string,
		sessionTimestamp: string,
		embedding: Buffer,
	): number {
		const result = this.db
			.prepare(
				`INSERT OR REPLACE INTO chunks
				(session_id, session_file, project, cwd, chunk_index, text, start_time, end_time, session_timestamp)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			)
			.run(sessionId, sessionFile, project, cwd, chunkIndex, text, startTime, endTime, sessionTimestamp);

		const chunkId = Number(result.lastInsertRowid);

		this.db.prepare("INSERT INTO chunks_vec (chunk_id, embedding) VALUES (CAST(? AS INTEGER), ?)").run(chunkId, embedding);

		return chunkId;
	}

	/**
	 * Mark a file as indexed.
	 */
	markFileIndexed(filePath: string, mtimeMs: number, size: number) {
		this.db
			.prepare(
				`INSERT OR REPLACE INTO indexed_files (file_path, mtime_ms, size, indexed_at)
				VALUES (?, ?, ?, ?)`,
			)
			.run(filePath, mtimeMs, size, new Date().toISOString());
	}

	/**
	 * Run operations in a transaction.
	 */
	transaction<T>(fn: () => T): T {
		return this.db.transaction(fn)();
	}

	/**
	 * Semantic vector search.
	 *
	 * Project filter uses normalized suffix matching so short names like "pi-ult"
	 * correctly match stored values like "--Volumes-4-GitHub-pi-ult--".
	 *
	 * Fetches a larger candidate pool when project/date filters are active to avoid
	 * returning too few results after post-filtering.
	 */
	vectorSearch(embedding: Buffer, limit: number, project?: string, after?: string, before?: string): SearchResult[] {
		const hasFilter = !!(project || after || before);
		const stmt = this.db.prepare(
			`SELECT c.*, v.distance
			FROM chunks_vec v
			JOIN chunks c ON c.id = v.chunk_id
			WHERE v.embedding MATCH ? AND k = ?
			ORDER BY v.distance`,
		);

		// When filters are active, expand k iteratively until we have enough results
		// or the vector index is exhausted (no new rows returned).
		let filtered: any[] = [];
		let k = hasFilter ? Math.max(limit * 20, 100) : limit;
		let prevTotal = -1;

		while (filtered.length < limit) {
			const rows = stmt.all(embedding, k) as any[];

			// Exhausted — no new rows even with larger k
			if (rows.length === prevTotal) break;
			prevTotal = rows.length;

			let candidates = rows;
			if (project) {
				candidates = candidates.filter((r: any) => projectMatches(r.project, project));
			}
			if (after) {
				candidates = candidates.filter((r: any) => r.session_timestamp && r.session_timestamp.slice(0, 10) >= after);
			}
			if (before) {
				candidates = candidates.filter((r: any) => r.session_timestamp && r.session_timestamp.slice(0, 10) <= before);
			}
			filtered = candidates;

			// If we got all available rows, stop expanding
			if (rows.length < k) break;

			// Expand k for next iteration
			k = k * 2;
		}

		filtered = filtered.slice(0, limit);

		return filtered.map((row: any) => ({
			id: row.id,
			sessionId: row.session_id,
			sessionFile: row.session_file,
			project: row.project,
			cwd: row.cwd,
			chunkIndex: row.chunk_index,
			text: row.text,
			startTime: row.start_time,
			endTime: row.end_time,
			sessionTimestamp: row.session_timestamp,
			distance: row.distance,
			score: 1 / (1 + row.distance),
		}));
	}

	/**
	 * Full-text search using LIKE with normalized project matching.
	 */
	textSearch(query: string, limit: number, project?: string, after?: string, before?: string): SearchResult[] {
		// If project filter given, collect matching raw project values first
		let projectValues: string[] | null = null;
		if (project) {
			const allProjects = (
				this.db.prepare("SELECT DISTINCT project FROM chunks").all() as { project: string }[]
			).map((r) => r.project);
			projectValues = allProjects.filter((p) => projectMatches(p, project));
		}

		let sql = "SELECT * FROM chunks WHERE text LIKE ?";
		const params: any[] = [`%${query}%`];

		if (projectValues !== null) {
			if (projectValues.length === 0) return []; // no matching projects
			sql += ` AND project IN (${projectValues.map(() => "?").join(",")})`;
			params.push(...projectValues);
		}
		if (after) {
			// date("") returns NULL in SQLite, so rows with empty timestamps are
			// excluded by the NULL comparison — correct behaviour.
			sql += " AND date(session_timestamp) >= ?";
			params.push(after);
		}
		if (before) {
			// date() extracts the date portion from ISO timestamp, inclusive of the day.
			// Rows with empty session_timestamp produce NULL and are excluded.
			sql += " AND date(session_timestamp) <= ?";
			params.push(before);
		}

		sql += " ORDER BY session_timestamp DESC LIMIT ?";
		params.push(limit);

		const rows = this.db.prepare(sql).all(...params) as any[];

		return rows.map((row) => ({
			id: row.id,
			sessionId: row.session_id,
			sessionFile: row.session_file,
			project: row.project,
			cwd: row.cwd,
			chunkIndex: row.chunk_index,
			text: row.text,
			startTime: row.start_time,
			endTime: row.end_time,
			sessionTimestamp: row.session_timestamp,
			distance: 0,
			score: 1,
		}));
	}

	/**
	 * Get index statistics.
	 */
	stats(): {
		totalChunks: number;
		totalSessions: number;
		totalFiles: number;
		projects: string[];
		oldestSession: string | null;
		newestSession: string | null;
	} {
		const totalChunks = (this.db.prepare("SELECT COUNT(*) as count FROM chunks").get() as any).count;
		const totalSessions = (
			this.db.prepare("SELECT COUNT(DISTINCT session_id) as count FROM chunks").get() as any
		).count;
		const totalFiles = (this.db.prepare("SELECT COUNT(*) as count FROM indexed_files").get() as any).count;
		const projects = (this.db.prepare("SELECT DISTINCT project FROM chunks ORDER BY project").all() as any[]).map(
			(r) => r.project,
		);
		const oldest = this.db.prepare("SELECT MIN(session_timestamp) as ts FROM chunks").get() as any;
		const newest = this.db.prepare("SELECT MAX(session_timestamp) as ts FROM chunks").get() as any;

		return {
			totalChunks,
			totalSessions,
			totalFiles,
			projects,
			oldestSession: oldest?.ts || null,
			newestSession: newest?.ts || null,
		};
	}

	close() {
		this.db.close();
	}
}
