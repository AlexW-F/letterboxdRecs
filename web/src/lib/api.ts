// Thin API client for the letterboxdRecs FastAPI backend.

export const API_BASE = (
	typeof window !== 'undefined' && (window as { LBRECS_API?: string }).LBRECS_API
) || import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export const TMDB_KEY = (import.meta.env.VITE_TMDB_KEY ?? '') as string;

export interface Explanation {
	top_contributing_rated_movies: [string, number][];
	dominant_genre_overlap: string;
	popularity_tier: string;
	source: string;
}

export interface Rec {
	movie_id: string;
	title: string;
	score: number;
	breakdown?: Record<string, number>;
	explanation?: Explanation;
}

export interface GroupRec {
	movie_id: string;
	title: string;
	score: number;
	per_member_score: Record<string, number>;
	fairness: number;
	explanation?: Explanation;
}

export interface UploadResult {
	hash: string;
	n_ratings_in: number;
	n_ratings_mapped: number;
	n_with_tmdb: number;
}

export interface Mode {
	name: string;
	description: string;
	weights: Record<string, number>;
}

export interface Strategy {
	name: string;
	description: string;
}

export interface PairwiseSimilarity {
	pair: [string, string];
	n_shared_movies: number;
	pearson_on_shared: number | null;
	cosine_on_taste: number | null;
}

export interface OverlapMovie {
	movie_id: string;
	title: string;
	member_ratings: Record<string, number>;
	mean: number;
	std: number;
}

export interface GroupAnalysis {
	member_names: string[];
	pairwise: PairwiseSimilarity[];
	consensus_movies: OverlapMovie[];
	disagreement_movies: OverlapMovie[];
}

export interface Health {
	status: string;
	svd_loaded: boolean;
	als_loaded: boolean;
	content_loaded: boolean;
	catalog_size: number;
	model_name: string;
	cache_dir: string;
}

async function jsonGet<T>(path: string): Promise<T> {
	const r = await fetch(`${API_BASE}${path}`);
	if (!r.ok) throw new Error(`${path} ${r.status}: ${await r.text()}`);
	return r.json() as Promise<T>;
}

async function jsonPost<T>(path: string, body: unknown): Promise<T> {
	const r = await fetch(`${API_BASE}${path}`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body)
	});
	if (!r.ok) throw new Error(`${path} ${r.status}: ${await r.text()}`);
	return r.json() as Promise<T>;
}

export async function getHealth(): Promise<Health> {
	return jsonGet<Health>('/health');
}

export async function getModes(): Promise<{ modes: Mode[] }> {
	return jsonGet<{ modes: Mode[] }>('/modes');
}

export async function getStrategies(): Promise<{ strategies: Strategy[] }> {
	return jsonGet<{ strategies: Strategy[] }>('/strategies');
}

export async function uploadLetterboxd(
	ratings: File,
	watched?: File | null
): Promise<UploadResult> {
	const fd = new FormData();
	fd.append('ratings', ratings);
	if (watched) fd.append('watched', watched);
	const r = await fetch(`${API_BASE}/upload-letterboxd`, { method: 'POST', body: fd });
	if (!r.ok) throw new Error(`upload ${r.status}: ${await r.text()}`);
	return r.json() as Promise<UploadResult>;
}

export interface IndividualRecRequest {
	hash: string;
	mode: string;
	top_n: number;
	exclude_rated?: boolean;
	exclude_watched?: boolean;
}

export interface IndividualRecResponse {
	hash: string;
	mode: string;
	n_ratings_used: number;
	n_watched_excluded: number;
	recommendations: Rec[];
}

export async function recommendIndividual(
	req: IndividualRecRequest
): Promise<IndividualRecResponse> {
	return jsonPost<IndividualRecResponse>('/recommend/individual', req);
}

export interface GroupRecRequest {
	hashes: string[];
	member_names?: string[];
	strategy: string;
	mode: string;
	top_n: number;
	exclude_rated?: boolean;
	exclude_watched?: boolean;
}

export interface GroupRecResponse {
	member_names: string[];
	strategy: string;
	mode: string;
	recommendations: GroupRec[];
}

export async function recommendGroup(req: GroupRecRequest): Promise<GroupRecResponse> {
	return jsonPost<GroupRecResponse>('/recommend/group', req);
}

export async function analyzeGroup(req: {
	hashes: string[];
	member_names?: string[];
	top_overlap?: number;
}): Promise<GroupAnalysis> {
	return jsonPost<GroupAnalysis>('/group/analyze', req);
}

// --- TMDB poster URL helper -----------------------------------------------
//
// The backend doesn't ship posters (they'd just be a wasted hop). The
// browser hits TMDB directly with a read-only API key. If VITE_TMDB_KEY is
// not set, we fall back to a generated placeholder so the UI still works
// at all.

const TMDB_LOOKUP_CACHE = new Map<string, Promise<string | null>>();

export function tmdbSearchURL(title: string, year?: number): string {
	const q = encodeURIComponent(title.replace(/\s*\((19|20)\d{2}\)\s*$/, '').trim());
	const yearParam = year ? `&primary_release_year=${year}` : '';
	return `https://api.themoviedb.org/3/search/movie?api_key=${TMDB_KEY}&query=${q}${yearParam}&include_adult=false`;
}

export function extractYear(title: string): number | undefined {
	const m = title.match(/\((19|20)(\d{2})\)\s*$/);
	if (!m) return undefined;
	return parseInt(m[1] + m[2], 10);
}

export async function tmdbPosterURL(title: string): Promise<string | null> {
	if (!TMDB_KEY) return null;
	if (TMDB_LOOKUP_CACHE.has(title)) return TMDB_LOOKUP_CACHE.get(title)!;
	const year = extractYear(title);
	const lookup = (async () => {
		try {
			const r = await fetch(tmdbSearchURL(title, year));
			if (!r.ok) return null;
			const data: { results: { poster_path: string | null }[] } = await r.json();
			if (!data.results?.length) return null;
			const poster = data.results[0]?.poster_path;
			return poster ? `https://image.tmdb.org/t/p/w342${poster}` : null;
		} catch {
			return null;
		}
	})();
	TMDB_LOOKUP_CACHE.set(title, lookup);
	return lookup;
}
