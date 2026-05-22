<script lang="ts">
	import type { Explanation } from '$lib/api';
	import { tmdbPosterURL } from '$lib/api';

	let {
		title,
		score,
		explanation = undefined,
		per_member_score = undefined,
		fairness = undefined,
		memberOrder = []
	}: {
		title: string;
		score: number;
		explanation?: Explanation;
		per_member_score?: Record<string, number>;
		fairness?: number;
		memberOrder?: string[];
	} = $props();

	let poster = $state<string | null>(null);
	let expanded = $state(false);

	$effect(() => {
		tmdbPosterURL(title).then((p) => (poster = p));
	});

	const popClass: Record<string, string> = {
		blockbuster: 'bg-amber-500/20 text-amber-200',
		popular: 'bg-sky-500/20 text-sky-200',
		niche: 'bg-purple-500/20 text-purple-200',
		obscure: 'bg-white/10 text-white/70'
	};

	function memberDot(score: number) {
		if (score >= 0.75) return 'bg-emerald-500';
		if (score >= 0.5) return 'bg-emerald-700';
		if (score >= 0.25) return 'bg-yellow-600';
		return 'bg-red-700';
	}

	const ordered = $derived(
		per_member_score
			? (memberOrder.length ? memberOrder : Object.keys(per_member_score))
					.filter((n) => n in per_member_score!)
					.map((n) => ({ name: n, score: per_member_score![n] }))
			: []
	);
</script>

<article class="rounded-lg bg-white/[0.03] border border-white/10 overflow-hidden flex">
	<div class="w-24 sm:w-28 flex-shrink-0 bg-black/40">
		{#if poster}
			<img src={poster} alt={title} class="w-full h-full object-cover" loading="lazy" />
		{:else}
			<div class="aspect-[2/3] w-full grid place-items-center text-2xl text-white/20">
				🎞️
			</div>
		{/if}
	</div>
	<div class="p-3 sm:p-4 flex-1 min-w-0">
		<div class="flex items-start gap-3">
			<h3 class="font-medium text-sm sm:text-base leading-snug flex-1 min-w-0">{title}</h3>
			<div class="text-right shrink-0">
				<div class="text-xl font-semibold tabular-nums">{score.toFixed(2)}</div>
				{#if fairness !== undefined}
					<div class="text-[10px] text-white/40">fair {fairness.toFixed(2)}</div>
				{/if}
			</div>
		</div>

		{#if explanation}
			<div class="mt-2 flex flex-wrap gap-1.5 items-center text-xs">
				{#if explanation.popularity_tier}
					<span
						class="px-1.5 py-0.5 rounded text-[10px] uppercase tracking-wider {popClass[
							explanation.popularity_tier
						] || 'bg-white/10 text-white/60'}"
					>
						{explanation.popularity_tier}
					</span>
				{/if}
				{#if explanation.dominant_genre_overlap}
					<span class="text-white/50">{explanation.dominant_genre_overlap}</span>
				{/if}
				{#if explanation.source}
					<span class="text-[10px] text-white/30">via {explanation.source}</span>
				{/if}
			</div>
		{/if}

		{#if ordered.length}
			<div class="mt-2 flex gap-2 items-center">
				{#each ordered as m (m.name)}
					<div class="flex items-center gap-1.5">
						<span class="inline-block w-2.5 h-2.5 rounded-full {memberDot(m.score)}"></span>
						<span class="text-[11px] text-white/70">{m.name}</span>
						<span class="text-[11px] text-white/40 tabular-nums">{m.score.toFixed(2)}</span>
					</div>
				{/each}
			</div>
		{/if}

		{#if explanation?.top_contributing_rated_movies?.length}
			<button
				type="button"
				class="mt-2 text-xs text-white/40 hover:text-white/80"
				onclick={() => (expanded = !expanded)}
			>
				{expanded ? 'hide' : 'why this?'}
			</button>
			{#if expanded}
				<div class="mt-1 text-xs text-white/60 space-y-0.5">
					Because you rated:
					<ul class="list-disc list-inside ml-1">
						{#each explanation.top_contributing_rated_movies as [t, r] (t)}
							<li>{t} <span class="text-white/40">— {r.toFixed(1)}★</span></li>
						{/each}
					</ul>
				</div>
			{/if}
		{/if}
	</div>
</article>
