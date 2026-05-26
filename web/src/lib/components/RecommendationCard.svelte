<script lang="ts">
	import { fly } from 'svelte/transition';
	import { ChevronDown, Star, Sparkles, ThumbsUp, Ban } from 'lucide-svelte';
	import Avatar from './Avatar.svelte';
	import type { Explanation, StreamingProvider } from '$lib/api';
	import { tmdbMovieMeta } from '$lib/api';

	let {
		rank = 0,
		movieId = undefined,
		title,
		score,
		explanation = undefined,
		per_member_score = undefined,
		fairness = undefined,
		memberOrder = [],
		breakdown = undefined,
		votes = undefined,
		voterName = undefined,
		onVote = undefined
	}: {
		rank?: number;
		movieId?: string;
		title: string;
		score: number;
		explanation?: Explanation;
		per_member_score?: Record<string, number>;
		fairness?: number;
		memberOrder?: string[];
		breakdown?: Record<string, number>;
		votes?: Record<string, 'up' | 'veto'>;
		voterName?: string;
		onVote?: (movieId: string, vote: 'up' | 'veto' | 'clear') => void;
	} = $props();

	const upCount = $derived(votes ? Object.values(votes).filter((v) => v === 'up').length : 0);
	const vetoCount = $derived(votes ? Object.values(votes).filter((v) => v === 'veto').length : 0);
	const myVote = $derived(voterName && votes ? votes[voterName] : undefined);
	const canVote = $derived(!!voterName && !!movieId && !!onVote);

	function clickVote(v: 'up' | 'veto') {
		if (!canVote) return;
		const next = myVote === v ? 'clear' : v;
		onVote!(movieId!, next);
	}

	// Signal-level decomposition. Skip rollup totals and zero contributions.
	const SIGNAL_ORDER = ['svd', 'als', 'content', 'implicit_bonus', 'popularity_penalty', 'diversity_penalty'];
	const SIGNAL_LABEL: Record<string, string> = {
		svd: 'SVD',
		als: 'ALS',
		content: 'genome',
		implicit_bonus: 'genre overlap',
		popularity_penalty: 'pop. penalty',
		diversity_penalty: 'diversity penalty'
	};
	const signals = $derived(
		breakdown
			? SIGNAL_ORDER.filter((k) => k in breakdown && Math.abs(breakdown[k]) > 0.005).map((k) => ({
					key: k,
					label: SIGNAL_LABEL[k] ?? k,
					value: breakdown[k]
				}))
			: []
	);
	const signalMaxAbs = $derived(
		signals.length ? Math.max(...signals.map((s) => Math.abs(s.value))) : 1
	);

	let poster = $state<string | null>(null);
	let providers = $state<StreamingProvider[]>([]);
	let expanded = $state(false);

	$effect(() => {
		tmdbMovieMeta(title).then((meta) => {
			poster = meta.posterUrl;
			providers = meta.providers;
		});
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
		{#if upCount > 0 || vetoCount > 0}
			<div
				class="absolute top-1.5 right-1.5 text-[10px] mono px-1.5 py-0.5 rounded flex items-center gap-1.5"
				style="background: rgba(10, 12, 16, 0.78); border: 1px solid var(--border);"
				title="{upCount} thumbs-up · {vetoCount} veto"
			>
				{#if upCount > 0}
					<span class="flex items-center gap-0.5" style="color: #6ee7b7;">
						<ThumbsUp size={9} />
						{upCount}
					</span>
				{/if}
				{#if vetoCount > 0}
					<span class="flex items-center gap-0.5" style="color: #f87171;">
						<Ban size={9} />
						{vetoCount}
					</span>
				{/if}
			</div>
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

		{#if providers.length}
			<div class="mt-2 flex flex-wrap items-center gap-1.5" title="Available to stream">
				{#each providers.slice(0, 5) as p (p.name)}
					<img
						src={p.logoUrl}
						alt={p.name}
						title={p.name}
						class="w-5 h-5 rounded-[5px] border"
						style="border-color: rgba(255,255,255,0.08); background: rgba(255,255,255,0.05);"
						loading="lazy"
					/>
				{/each}
				{#if providers.length > 5}
					<span class="text-[10px] mono" style="color: var(--ink-faint);">
						+{providers.length - 5}
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

		{#if explanation?.top_contributing_rated_movies?.length || signals.length}
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
					class="mt-2 text-xs rounded-md p-2.5 space-y-2.5"
					style="background: rgba(0,0,0,0.3); border: 1px solid var(--border); color: var(--ink-muted);"
					transition:fly={{ y: -4, duration: 200 }}
				>
					{#if signals.length}
						<div>
							<div class="flex items-center gap-1.5 mb-1.5" style="color: var(--ink-dim);">
								<Sparkles size={11} />
								what lifted this onto the list:
							</div>
							<div class="space-y-1">
								{#each signals as s (s.key)}
									{@const pct = (Math.abs(s.value) / signalMaxAbs) * 100}
									{@const isNeg = s.value < 0}
									<div class="flex items-center gap-2">
										<span class="w-24 shrink-0 text-[10px] mono" style="color: var(--ink-faint);">
											{s.label}
										</span>
										<div class="flex-1 relative h-1.5 rounded-sm" style="background: rgba(255,255,255,0.05);">
											<div
												class="absolute top-0 bottom-0 rounded-sm"
												class:bg-emerald-500={!isNeg}
												class:bg-rose-500={isNeg}
												style="width: {pct.toFixed(1)}%; opacity: 0.8; {isNeg ? 'right: 50%;' : 'left: 50%;'} max-width: 50%;"
											></div>
											<div
												class="absolute top-0 bottom-0 w-px"
												style="left: 50%; background: rgba(255,255,255,0.15);"
											></div>
										</div>
										<span class="w-10 text-right shrink-0 text-[10px] mono tabular-nums" style="color: var(--ink-faint);">
											{s.value >= 0 ? '+' : ''}{s.value.toFixed(2)}
										</span>
									</div>
								{/each}
							</div>
						</div>
					{/if}
					{#if explanation?.top_contributing_rated_movies?.length}
						<div>
							<div class="flex items-center gap-1.5 mb-1" style="color: var(--ink-dim);">
								<Star size={11} fill="currentColor" style="color: var(--accent);" />
								closest matches from your ratings:
							</div>
							<div class="space-y-1">
								{#each explanation.top_contributing_rated_movies as [t, r] (t)}
									<div class="flex items-center gap-2">
										<Star size={10} fill="currentColor" style="color: var(--accent);" />
										<span class="flex-1">{t}</span>
										<span class="mono text-[10px]" style="color: var(--ink-faint);">{r.toFixed(1)}★</span>
									</div>
								{/each}
							</div>
						</div>
					{/if}
				</div>
			{/if}
		{/if}

		{#if canVote}
			<div class="mt-3 flex items-center gap-1.5 text-xs">
				<button
					type="button"
					class="btn btn-pill text-xs px-2.5 py-1"
					style={myVote === 'up'
						? 'background: rgba(110, 231, 183, 0.18); border: 1px solid rgba(110, 231, 183, 0.55); color: #6ee7b7;'
						: 'background: var(--surface); border: 1px solid var(--border); color: var(--ink-muted);'}
					onclick={() => clickVote('up')}
					title="Vote up as {voterName}"
				>
					<ThumbsUp size={11} />
					yes
				</button>
				<button
					type="button"
					class="btn btn-pill text-xs px-2.5 py-1"
					style={myVote === 'veto'
						? 'background: rgba(248, 113, 113, 0.16); border: 1px solid rgba(248, 113, 113, 0.5); color: #fca5a5;'
						: 'background: var(--surface); border: 1px solid var(--border); color: var(--ink-muted);'}
					onclick={() => clickVote('veto')}
					title="Veto as {voterName}"
				>
					<Ban size={11} />
					skip
				</button>
				<span class="ml-1 text-[10px]" style="color: var(--ink-faint);">
					voting as <strong style="color: var(--ink-muted);">{voterName}</strong>
				</span>
			</div>
		{/if}
	</div>
</article>
