<script lang="ts">
	import FileDropzone from '$lib/components/FileDropzone.svelte';
	import ModeSelector from '$lib/components/ModeSelector.svelte';
	import RecommendationCard from '$lib/components/RecommendationCard.svelte';
	import {
		getModes,
		recommendIndividual,
		uploadLetterboxd,
		type IndividualRecResponse,
		type Mode
	} from '$lib/api';

	let modes = $state<Mode[]>([]);
	let mode = $state('balanced');
	let topN = $state(10);
	let excludeRated = $state(true);
	let excludeWatched = $state(true);

	let ratingsFile = $state<File | null>(null);
	let watchedFile = $state<File | null>(null);
	let hash = $state<string | null>(null);
	let uploadInfo = $state<{ in: number; mapped: number; with_tmdb: number } | null>(null);

	let busy = $state(false);
	let error = $state<string | null>(null);
	let result = $state<IndividualRecResponse | null>(null);

	$effect(() => {
		getModes()
			.then((m) => (modes = m.modes))
			.catch(() => (modes = []));
	});

	async function uploadAndRecommend() {
		error = null;
		if (!ratingsFile) {
			error = 'Drop in ratings.csv';
			return;
		}
		busy = true;
		try {
			if (!hash) {
				const u = await uploadLetterboxd(ratingsFile, watchedFile);
				hash = u.hash;
				uploadInfo = { in: u.n_ratings_in, mapped: u.n_ratings_mapped, with_tmdb: u.n_with_tmdb };
			}
			result = await recommendIndividual({
				hash: hash!,
				mode,
				top_n: topN,
				exclude_rated: excludeRated,
				exclude_watched: excludeWatched
			});
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			busy = false;
		}
	}

	async function rerecommend() {
		if (!hash) {
			await uploadAndRecommend();
			return;
		}
		busy = true;
		try {
			result = await recommendIndividual({
				hash,
				mode,
				top_n: topN,
				exclude_rated: excludeRated,
				exclude_watched: excludeWatched
			});
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			busy = false;
		}
	}

	function reset() {
		hash = null;
		uploadInfo = null;
		result = null;
		ratingsFile = null;
		watchedFile = null;
	}
</script>

<section class="space-y-6">
	<header class="flex items-center justify-between">
		<div>
			<h1 class="text-2xl font-semibold tracking-tight">Solo recommendations</h1>
			<p class="text-sm text-white/50">
				Upload your Letterboxd export, get top-N picks across four modes.
			</p>
		</div>
		<a href="/" class="text-sm text-sky-300 hover:text-sky-200">← group flow</a>
	</header>

	{#if !hash}
		<div class="rounded-lg bg-white/5 border border-white/10 p-4 space-y-3 max-w-lg">
			<FileDropzone
				label="ratings.csv (TMDB-enriched)"
				bind:file={ratingsFile}
				hint="with tmdb_id column"
			/>
			<FileDropzone label="watched.csv (optional)" bind:file={watchedFile} />
			{#if error}
				<p class="text-sm text-red-400">{error}</p>
			{/if}
			<button
				type="button"
				onclick={uploadAndRecommend}
				disabled={busy}
				class="w-full px-4 py-2 rounded bg-emerald-500 text-emerald-950 font-medium hover:bg-emerald-400 disabled:bg-emerald-700/50"
			>
				{busy ? 'working…' : 'Upload + recommend'}
			</button>
		</div>
	{:else}
		<div class="rounded-lg bg-white/5 border border-white/10 p-4 space-y-4">
			<div class="flex items-start justify-between">
				<div>
					<p class="text-sm text-white/70">
						{#if uploadInfo}
							{uploadInfo.mapped} of {uploadInfo.in} ratings mapped to ml-32m ({uploadInfo.with_tmdb}
							had a TMDB ID).
						{/if}
					</p>
					<p class="text-xs text-white/30">hash {hash.slice(0, 16)}…</p>
				</div>
				<button
					type="button"
					onclick={reset}
					class="text-xs text-white/40 hover:text-white"
				>
					change upload
				</button>
			</div>

			<ModeSelector {modes} bind:value={mode} />

			<div class="flex flex-wrap items-center gap-4 text-sm">
				<label class="flex items-center gap-2">
					<span class="text-white/50 text-xs uppercase tracking-wider">top n</span>
					<input
						type="number"
						min="1"
						max="50"
						bind:value={topN}
						class="w-16 bg-black/30 border border-white/10 rounded px-2 py-1 text-sm"
					/>
				</label>
				<label class="flex items-center gap-2 text-white/70">
					<input type="checkbox" bind:checked={excludeRated} />
					exclude rated
				</label>
				<label class="flex items-center gap-2 text-white/70">
					<input type="checkbox" bind:checked={excludeWatched} />
					exclude watched
				</label>
				<button
					type="button"
					onclick={rerecommend}
					disabled={busy}
					class="ml-auto px-4 py-1.5 rounded bg-sky-500 text-sky-950 font-medium hover:bg-sky-400 disabled:opacity-50"
				>
					{busy ? 'recomputing…' : 'recompute'}
				</button>
			</div>
		</div>

		{#if result}
			<div class="grid grid-cols-1 md:grid-cols-2 gap-3">
				{#each result.recommendations as r (r.movie_id)}
					<RecommendationCard
						title={r.title}
						score={r.score}
						explanation={r.explanation}
					/>
				{/each}
			</div>
		{/if}
	{/if}
</section>
