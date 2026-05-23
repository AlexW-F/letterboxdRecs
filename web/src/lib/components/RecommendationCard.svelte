<script lang="ts">
	import { fly } from 'svelte/transition';
	import { ChevronDown, Star, Sparkles } from 'lucide-svelte';
	import Avatar from './Avatar.svelte';
	import type { Explanation } from '$lib/api';
	import { tmdbPosterURL } from '$lib/api';

	let {
		rank = 0,
		title,
		score,
		explanation = undefined,
		per_member_score = undefined,
		fairness = undefined,
		memberOrder = []
	}: {
		rank?: number;
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
		blockbuster: 'chip-accent',
		popular: 'chip-violet',
		niche: 'chip-brand',
		obscure: 'chip'
	};

	function memberDotClass(score: number): string {
		if (score >= 0.75) return 'bg-emerald-400';
		if (score >= 0.5) return 'bg-emerald-600';
		if (score >= 0.25) return 'bg-amber-500';
		return 'bg-rose-500';
	}

	const ordered = $derived(
		per_member_score
			? (memberOrder.length ? memberOrder : Object.keys(per_member_score))
					.filter((n) => n in per_member_score!)
					.map((n) => ({ name: n, score: per_member_score![n] }))
			: []
	);

	const titleClean = $derived(title.replace(/\s*\((19|20)\d{2}\)\s*$/, ''));
	const yearMatch = $derived(title.match(/\((19|20)\d{2}\)\s*$/));
</script>

<article
	class="surface card-hover overflow-hidden flex group relative"
	in:fly={{ y: 12, delay: Math.min(rank * 25, 250), duration: 320 }}
>
	<div
		class="w-24 sm:w-32 flex-shrink-0 relative overflow-hidden"
		style="background: linear-gradient(180deg, #11141a, #0a0c10);"
	>
		{#if poster}
			<img
				src={poster}
				alt={title}
				class="w-full h-full object-cover transition duration-500 group-hover:scale-105"
				loading="lazy"
			/>
			<div
				class="absolute inset-0 pointer-events-none"
				style="background: linear-gradient(180deg, transparent 50%, rgba(10, 12, 16, 0.7) 100%);"
			></div>
		{:else}
			<div class="aspect-[2/3] grid place-items-center text-3xl" style="color: var(--ink-faint);">
				🎞️
			</div>
		{/if}
		{#if rank > 0}
			<span
				class="absolute top-1.5 left-1.5 text-[10px] font-semibold tabular-nums px-1.5 py-0.5 rounded mono"
				style="background: rgba(10, 12, 16, 0.78); border: 1px solid var(--border); color: var(--ink-muted);"
			>
				#{rank}
			</span>
		{/if}
	</div>

	<div class="p-3 sm:p-4 flex-1 min-w-0">
		<div class="flex items-start gap-3">
			<div class="flex-1 min-w-0">
				<h3 class="font-medium leading-snug text-balance" style="font-size: 0.95rem;">
					{titleClean}
				</h3>
				{#if yearMatch}
					<div class="text-[11px] mono mt-0.5" style="color: var(--ink-faint);">
						{yearMatch[0].replace(/[()]/g, '').trim()}
					</div>
				{/if}
			</div>
			<div class="text-right shrink-0">
				<div
					class="text-2xl font-semibold tabular-nums leading-none"
					style="background: linear-gradient(135deg, #6ee7b7, #34d399 60%, #fbbf24); background-clip: text; -webkit-background-clip: text; color: transparent;"
				>
					{score.toFixed(2)}
				</div>
				{#if fairness !== undefined}
					<div class="text-[10px] mono mt-1" style="color: var(--ink-faint);">
						fair {fairness.toFixed(2)}
					</div>
				{/if}
			</div>
		</div>

		{#if explanation}
			<div class="mt-2 flex flex-wrap gap-1.5 items-center text-xs">
				{#if explanation.popularity_tier}
					<span class="chip {popClass[explanation.popularity_tier] || ''}">
						{explanation.popularity_tier}
					</span>
				{/if}
				{#if explanation.dominant_genre_overlap}
					<span style="color: var(--ink-muted);">{explanation.dominant_genre_overlap}</span>
				{/if}
				{#if explanation.source}
					<span class="text-[10px] mono ml-auto" style="color: var(--ink-faint);">
						via {explanation.source}
					</span>
				{/if}
			</div>
		{/if}

		{#if ordered.length > 0}
			<div class="mt-3 flex flex-wrap gap-x-3 gap-y-1.5 items-center">
				{#each ordered as m (m.name)}
					<div class="flex items-center gap-1.5">
						<Avatar name={m.name} size={20} />
						<span class="text-[12px]">{m.name}</span>
						<div class="flex items-center gap-0.5">
							{#each [0, 1, 2, 3, 4] as i (i)}
								<span
									class="w-1 h-3 rounded-sm transition"
									class:opacity-40={m.score < (i + 1) * 0.2}
									class:bg-emerald-400={m.score >= (i + 1) * 0.2 && m.score >= 0.75}
									class:bg-emerald-700={m.score >= (i + 1) * 0.2 && m.score >= 0.5 && m.score < 0.75}
									class:bg-amber-500={m.score >= (i + 1) * 0.2 && m.score >= 0.25 && m.score < 0.5}
									class:bg-rose-500={m.score >= (i + 1) * 0.2 && m.score < 0.25}
									style={m.score < (i + 1) * 0.2 ? 'background: rgba(255,255,255,0.08);' : ''}
								></span>
							{/each}
						</div>
					</div>
				{/each}
			</div>
		{/if}

		{#if explanation?.top_contributing_rated_movies?.length}
			<button
				class="mt-2 text-xs hover:underline inline-flex items-center gap-1"
				style="color: var(--ink-dim);"
				onclick={() => (expanded = !expanded)}
			>
				<ChevronDown
					size={12}
					style="transition: transform 200ms; {expanded ? 'transform: rotate(180deg);' : ''}"
				/>
				{expanded ? 'hide why' : 'why this?'}
			</button>
			{#if expanded}
				<div
					class="mt-2 text-xs rounded-md p-2.5 space-y-1"
					style="background: rgba(0,0,0,0.3); border: 1px solid var(--border); color: var(--ink-muted);"
					transition:fly={{ y: -4, duration: 200 }}
				>
					<div class="flex items-center gap-1.5 mb-1" style="color: var(--ink-dim);">
						<Sparkles size={11} />
						closest matches from your ratings:
					</div>
					{#each explanation.top_contributing_rated_movies as [t, r] (t)}
						<div class="flex items-center gap-2">
							<Star size={10} fill="currentColor" style="color: var(--accent);" />
							<span class="flex-1">{t}</span>
							<span class="mono text-[10px]" style="color: var(--ink-faint);">{r.toFixed(1)}★</span>
						</div>
					{/each}
				</div>
			{/if}
		{/if}
	</div>
</article>
