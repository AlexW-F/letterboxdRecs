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
	breakdown?: Record<string, number>;
}

export interface UploadResult {
	hash: string;
	n_ratings_in: number;
	n_ratings_mapped: number;
	n_with_tmdb: number;
	n_watchlist: number;
	source?: 'csv' | 'letterboxd_rss';
	letterboxd_username?: string | null;
}

export interface WatchlistOverlapItem {
	movie_id: string;
	title: string;
	members: string[];
	count: number;
}

export interface WatchlistOverlapResponse {
	member_names: string[];
	n_with_watchlist: number;
	items: WatchlistOverlapItem[];
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
	movie_space_loaded: boolean;
}

export interface BackgroundScatter {
	n: number;
	coords: [number, number, number][];
	titles: string[];
	genres: string[];
	popularity: number[];
	genre_colors: Record<string, string>;
}

export async function fetchBackgroundCoords(limit = 3000): Promise<BackgroundScatter> {
	const r = await fetch(`${API_BASE}/explore/background?limit=${limit}`);
	if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
	return r.json() as Promise<BackgroundScatter>;
}

export async function fetchPersonalizedVizHTML(hash: string, label?: string): Promise<string> {
	const params = new URLSearchParams({ hash });
	if (label) params.set('label', label);
	const r = await fetch(`${API_BASE}/explore/personalized?${params}`);
	if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
	return r.text();
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
	watched?: File | null,
	watchlist?: File | null
): Promise<UploadResult> {
	const fd = new FormData();
	fd.append('ratings', ratings);
	if (watched) fd.append('watched', watched);
	if (watchlist) fd.append('watchlist', watchlist);
	const r = await fetch(`${API_BASE}/upload-letterboxd`, { method: 'POST', body: fd });
	if (!r.ok) throw new Error(`upload ${r.status}: ${await r.text()}`);
	return r.json() as Promise<UploadResult>;
}

// --- Shared groups (multi-device join link + voting) --------------------

export interface GroupMember {
	name: string;
	hash: string;
	n_ratings_mapped: number;
	n_watchlist: number;
	source: string;
	letterboxd_username?: string | null;
	joined_at: number;
}

export interface GroupState {
	group_id: string;
	created_at: number;
	members: GroupMember[];
	votes: Record<string, Record<string, 'up' | 'veto'>>;
}

export async function createGroup(name?: string): Promise<GroupState> {
	return jsonPost<GroupState>('/group', { name: name ?? null });
}

export async function createDemoGroup(): Promise<GroupState> {
	return jsonPost<GroupState>('/group/demo', {});
}

export async function getGroup(groupId: string): Promise<GroupState> {
	return jsonGet<GroupState>(`/group/${encodeURIComponent(groupId)}`);
}

export async function joinGroup(
	groupId: string,
	body: { name: string; hash: string }
): Promise<GroupState> {
	return jsonPost<GroupState>(`/group/${encodeURIComponent(groupId)}/join`, body);
}

export async function leaveGroup(groupId: string, memberName: string): Promise<GroupState> {
	return jsonPost<GroupState>(
		`/group/${encodeURIComponent(groupId)}/leave/${encodeURIComponent(memberName)}`,
		{}
	);
}

export async function castGroupVote(
	groupId: string,
	body: { member_name: string; movie_id: string; vote: 'up' | 'veto' | 'clear' }
): Promise<GroupState> {
	return jsonPost<GroupState>(`/group/${encodeURIComponent(groupId)}/vote`, body);
}

export async function uploadLetterboxdUsername(username: string): Promise<UploadResult> {
	const r = await fetch(`${API_BASE}/upload-letterboxd-username`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ username })
	});
	if (!r.ok) {
		// Backend returns {detail: "..."} for HTTPException; surface that string.
		let msg = `${r.status}`;
		try {
			const body = await r.json();
			msg = body.detail || msg;
		} catch {
			msg = await r.text();
		}
		throw new Error(msg);
	}
	return r.json() as Promise<UploadResult>;
}

export async function groupWatchlistOverlap(req: {
	hashes: string[];
	member_names?: string[];
	min_members?: number;
	top_n?: number;
}): Promise<WatchlistOverlapResponse> {
	return jsonPost<WatchlistOverlapResponse>('/group/watchlist-overlap', req);
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
	exclude_seen_by_any?: boolean;
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

export async function recommendGroupDisagreement(
	req: GroupRecRequest
): Promise<GroupRecResponse> {
	return jsonPost<GroupRecResponse>('/recommend/group/disagreement', req);
}

export async function analyzeGroup(req: {
	hashes: string[];
	member_names?: string[];
	top_overlap?: number;
}): Promise<GroupAnalysis> {
	return jsonPost<GroupAnalysis>('/group/analyze', req);
}

// --- TMDB poster + watch-provider helpers --------------------------------
//
// The backend doesn't ship posters or streaming info (a wasted hop — the
// browser can hit TMDB directly with a read-only API key). If
// VITE_TMDB_KEY is not set, helpers return null/empty and the UI degrades.

export interface StreamingProvider {
	name: string;
	logoUrl: string;
	priority: number;
}

export interface MovieMeta {
	tmdbId: number | null;
	posterUrl: string | null;
	providers: StreamingProvider[]; // subscription/flatrate only — what you can stream without paying again
}

const META_CACHE = new Map<string, Promise<MovieMeta>>();
const EMPTY_META: MovieMeta = { tmdbId: null, posterUrl: null, providers: [] };

function tmdbKey(): string {
	if (typeof window !== 'undefined') {
		const override = (window as { LBRECS_TMDB?: string }).LBRECS_TMDB;
		if (override) return override;
	}
	return TMDB_KEY;
}

export function tmdbSearchURL(title: string, year?: number): string {
	const q = encodeURIComponent(title.replace(/\s*\((19|20)\d{2}\)\s*$/, '').trim());
	const yearParam = year ? `&primary_release_year=${year}` : '';
	return `https://api.themoviedb.org/3/search/movie?api_key=${tmdbKey()}&query=${q}${yearParam}&include_adult=false`;
}

export function extractYear(title: string): number | undefined {
	const m = title.match(/\((19|20)(\d{2})\)\s*$/);
	if (!m) return undefined;
	return parseInt(m[1] + m[2], 10);
}

// Region defaults to US; flip via localStorage('lbrecs_region') if needed.
function watchRegion(): string {
	if (typeof window === 'undefined') return 'US';
	return (window.localStorage?.getItem('lbrecs_region') || 'US').toUpperCase();
}

export async function tmdbMovieMeta(title: string): Promise<MovieMeta> {
	if (META_CACHE.has(title)) return META_CACHE.get(title)!;

	const lookup = (async (): Promise<MovieMeta> => {
		const key = tmdbKey();
		if (!key) return EMPTY_META;
		try {
			const year = extractYear(title);
			const searchRes = await fetch(tmdbSearchURL(title, year));
			if (!searchRes.ok) return EMPTY_META;
			const searchData: { results: { id: number; poster_path: string | null }[] } =
				await searchRes.json();
			const hit = searchData.results?.[0];
			if (!hit) return EMPTY_META;
			const posterUrl = hit.poster_path ? `https://image.tmdb.org/t/p/w342${hit.poster_path}` : null;

			// Watch-providers is a second call but TMDB's free tier handles it.
			let providers: StreamingProvider[] = [];
			try {
				const provRes = await fetch(
					`https://api.themoviedb.org/3/movie/${hit.id}/watch/providers?api_key=${tmdbKey()}`
				);
				if (provRes.ok) {
					const provData: {
						results: Record<
							string,
							{
								flatrate?: { provider_name: string; logo_path: string; display_priority: number }[];
							}
						>;
					} = await provRes.json();
					const region = watchRegion();
					const flat = provData.results?.[region]?.flatrate ?? [];
					providers = flat
						.slice()
						.sort((a, b) => a.display_priority - b.display_priority)
						.map((p) => ({
							name: p.provider_name,
							logoUrl: `https://image.tmdb.org/t/p/w45${p.logo_path}`,
							priority: p.display_priority
						}));
				}
			} catch {
				// Provider lookup is best-effort
			}

			return { tmdbId: hit.id, posterUrl, providers };
		} catch {
			return EMPTY_META;
		}
	})();
	META_CACHE.set(title, lookup);
	return lookup;
}

// Back-compat shim so other pages (explore, etc.) that just want the poster
// keep working unchanged.
export async function tmdbPosterURL(title: string): Promise<string | null> {
	const meta = await tmdbMovieMeta(title);
	return meta.posterUrl;
}
