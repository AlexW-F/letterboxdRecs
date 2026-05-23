<script lang="ts">
	import { fly } from 'svelte/transition';
	import { Loader2, Wand2, Upload, RefreshCw, RotateCcw } from 'lucide-svelte';
	import FileDropzone from '$lib/components/FileDropzone.svelte';
	import ModeSelector from '$lib/components/ModeSelector.svelte';
	import RecommendationCard from '$lib/components/RecommendationCard.svelte';
	import Stat from '$lib/components/Stat.svelte';
	import Shimmer from '$lib/components/Shimmer.svelte';
	import {
		getModes,
		recommendIndividual,
		uploadLetterboxd,
		type IndividualRecResponse,
		type Mode
	} from '$lib/api';

	let modes = $state<Mode[]>([]);
	let mode = $state('balanced');
	let topN = $state(12);
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

<section class="space-y-6 anim-fade-up">
	<header class="flex items-end justify-between flex-wrap gap-3">
		<div>
			<span class="chip chip-brand">Solo · personalized re-rank</span>
			<h1 class="display-md mt-2" style="font-family: 'Instrument Serif', Georgia, serif; font-style: italic;">
				Recommendations <span class="text-gradient">for one</span>.
			</h1>
			<p class="text-sm" style="color: var(--ink-muted);">
				Upload your Letterboxd ratings, get top-N across four modes. Pick a tone, adjust the
				exclusion toggles, recompute.
			</p>
		</div>
		<a
			href="/"
			class="btn btn-ghost btn-pill text-xs"
		>← group flow</a>
	</header>

	{#if !hash}
		<div class="surface p-5 space-y-4 max-w-xl anim-fade-up">
			<div class="flex items-center gap-2">
				<Upload size={16} style="color: var(--brand);" />
				<h2 class="font-medium">Drop your Letterboxd CSVs</h2>
			</div>
			<FileDropzone
				label="ratings.csv"
				bind:file={ratingsFile}
				hint="from your Letterboxd export — TMDB enrichment server-side"
			/>
			<FileDropzone
				label="watched.csv (optional)"
				bind:file={watchedFile}
				hint="excludes watched-but-unrated films from recs"
			/>
			{#if error}
				<p
					class="text-sm rounded-md px-3 py-2"
					style="background: var(--rose-dim); border: 1px solid rgba(248, 113, 113, 0.3); color: #fecaca;"
				>
					{error}
				</p>
			{/if}
			<button
				class="btn btn-primary w-full"
				onclick={uploadAndRecommend}
				disabled={busy}
			>
				{#if busy}
					<Loader2 size={16} class="animate-spin" />
					Uploading + enriching + recommending…
				{:else}
					<Wand2 size={16} />
					Upload + recommend
				{/if}
			</button>
			{#if busy}
				<p class="text-xs" style="color: var(--ink-dim);">
					~15s for the first TMDB enrichment of ~250 ratings. Instant on re-upload.
				</p>
			{/if}
		</div>
	{:else}
		<div class="grid grid-cols-2 md:grid-cols-4 gap-3 anim-fade-up">
			<Stat
				label="ratings uploaded"
				value={uploadInfo?.in ?? '—'}
				hint={uploadInfo ? `${uploadInfo.with_tmdb} had TMDB IDs` : ''}
			/>
			<Stat
				label="mapped to ml-32m"
				value={uploadInfo?.mapped ?? '—'}
				hint={uploadInfo ? `${Math.round(((uploadInfo.mapped ?? 0) / (uploadInfo.in || 1)) * 100)}% coverage` : ''}
			/>
			<Stat
				label="mode"
				value={mode}
				hint={modes.find((m) => m.name === mode)?.description.slice(0, 36) ?? ''}
			/>
			<Stat label="top n" value={topN} hint="cards rendered below" />
		</div>

		<div class="surface p-5 space-y-4 anim-fade-up" style="animation-delay: 80ms;">
			<ModeSelector {modes} bind:value={mode} />

			<div class="flex flex-wrap items-center gap-4 text-sm pt-1">
				<label class="flex items-center gap-2">
					<span class="text-[10px] uppercase tracking-[0.14em]" style="color: var(--ink-faint);">
						top n
					</span>
					<input
						type="number"
						min="1"
						max="50"
						bind:value={topN}
						class="input"
						style="width: 5rem; padding: 0.3rem 0.5rem;"
					/>
				</label>
				<label class="flex items-center gap-2 cursor-pointer">
					<input type="checkbox" bind:checked={excludeRated} />
					<span style="color: var(--ink-muted);">exclude rated</span>
				</label>
				<label class="flex items-center gap-2 cursor-pointer">
					<input type="checkbox" bind:checked={excludeWatched} />
					<span style="color: var(--ink-muted);">exclude watched</span>
				</label>

				<div class="ml-auto flex gap-2">
					<button class="btn btn-ghost btn-pill text-xs" onclick={reset}>
						<RotateCcw size={12} />
						change upload
					</button>
					<button class="btn btn-secondary" onclick={rerecommend} disabled={busy}>
						{#if busy}
							<Loader2 size={14} class="animate-spin" />
							recomputing
						{:else}
							<RefreshCw size={14} />
							recompute
						{/if}
					</button>
				</div>
			</div>
		</div>

		{#if busy && !result}
			<div class="grid grid-cols-1 md:grid-cols-2 gap-3 anim-fade-up">
				{#each Array(6) as _, i (i)}
					<div class="surface p-3 flex gap-3 h-32">
						<Shimmer width="6rem" height="100%" rounded="0.5rem" />
						<div class="flex-1 space-y-2 py-1">
							<Shimmer height="1rem" width="70%" />
							<Shimmer height="0.75rem" width="40%" />
							<Shimmer height="2rem" width="100%" />
						</div>
					</div>
				{/each}
			</div>
		{/if}

		{#if result}
			<div class="grid grid-cols-1 md:grid-cols-2 gap-3">
				{#each result.recommendations as r, i (r.movie_id)}
					<RecommendationCard
						rank={i + 1}
						title={r.title}
						score={r.score}
						explanation={r.explanation}
					/>
				{/each}
			</div>
		{/if}
	{/if}
</section>
