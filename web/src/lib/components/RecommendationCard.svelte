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
		blockbuster: 'chip-brand',
		popular: 'chip-violet',
		niche: 'chip-rose',
		obscure: 'chip'
	};

	// Warm-band member fit: moss when it fits, rust when it doesn't.
	function memberBarColor(s: number): string {
		if (s >= 0.75) return '#8faf7a';
		if (s >= 0.5) return '#78824b';
		if (s >= 0.25) return '#c9a554';
		return '#bb7744';
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
	class="surface card-hover overflow-hidden flex group relative {rank % 2 === 1 ? 'tilt-a' : 'tilt-b'}"
	style="background: rgba(26, 24, 20, 0.9);"
	in:fly={{ y: 12, delay: Math.min(rank * 25, 250), duration: 320 }}
>
	<!-- Tear-off ticket strip -->
	{#if rank > 0}
		<div
			class="stub-tear w-9 flex-shrink-0 flex flex-col items-center justify-center gap-2 py-3"
		>
			<span
				class="board text-[9px]"
				style="color: var(--ink-faint); writing-mode: vertical-rl; letter-spacing: 0.22em;"
			>
				Admit&nbsp;all
			</span>
			<span class="mono text-[10px]" style="color: var(--ink-muted);">
				№{String(rank).padStart(3, '0')}
			</span>
		</div>
	{/if}

	<div
		class="w-20 sm:w-28 flex-shrink-0 relative overflow-hidden"
		style="background: linear-gradient(180deg, #242219, #171613);"
	>
		{#if poster}
			<img
				src={poster}
				alt={title}
				class="w-full h-full object-cover transition duration-500 group-hover:scale-105"
				loading="lazy"
				onerror={() => (poster = null)}
			/>
			<div
				class="absolute inset-0 pointer-events-none"
				style="background: linear-gradient(180deg, transparent 50%, rgba(23, 22, 19, 0.7) 100%);"
			></div>
		{:else}
			<div class="aspect-[2/3] grid place-items-center text-3xl" style="color: var(--ink-faint);">
				🎞️
			</div>
		{/if}
		{#if upCount > 0 || vetoCount > 0}
			<div
				class="absolute top-1.5 right-1.5 text-[10px] mono px-1.5 py-0.5 rounded flex items-center gap-1.5"
				style="background: rgba(23, 22, 19, 0.82); border: 1px solid var(--border);"
				title="{upCount} thumbs-up · {vetoCount} veto"
			>
				{#if upCount > 0}
					<span class="flex items-center gap-0.5" style="color: var(--green);">
						<ThumbsUp size={9} />
						{upCount}
					</span>
				{/if}
				{#if vetoCount > 0}
					<span class="flex items-center gap-0.5" style="color: var(--rust);">
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
				<h3
					class="leading-snug text-balance board font-medium"
					style="font-size: 1.02rem; letter-spacing: 0.06em;"
				>
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
					class="text-2xl font-medium tabular-nums leading-none board"
					style="color: var(--brand); text-shadow: 0 0 10px rgba(201, 165, 84, 0.45);"
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
						style="border-color: rgba(215, 196, 131, 0.14); background: rgba(215, 196, 131, 0.06);"
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
									style={m.score >= (i + 1) * 0.2
										? `background: ${memberBarColor(m.score)};`
										: 'background: rgba(215, 196, 131, 0.1);'}
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
				aria-expanded={expanded}
				onclick={() => (expanded = !expanded)}
			>
				<ChevronDown
					size={12}
					style="transition: transform 200ms; {expanded ? 'transform: rotate(180deg);' : ''}"
				/>
				{expanded ? 'hide the notes' : 'projectionist’s notes'}
			</button>
			{#if expanded}
				<div
					class="mt-2 text-xs rounded-md p-2.5 space-y-2.5"
					style="background: rgba(12, 11, 9, 0.45); border: 1px solid var(--border); color: var(--ink-muted);"
					transition:fly={{ y: -4, duration: 200 }}
				>
					{#if signals.length}
						<div>
							<div class="flex items-center gap-1.5 mb-1.5" style="color: var(--ink-dim);">
								<Sparkles size={11} />
								what lifted this onto the board:
							</div>
							<div class="space-y-1">
								{#each signals as s (s.key)}
									{@const pct = (Math.abs(s.value) / signalMaxAbs) * 100}
									{@const isNeg = s.value < 0}
									<div class="flex items-center gap-2">
										<span class="w-24 shrink-0 text-[10px] mono" style="color: var(--ink-faint);">
											{s.label}
										</span>
										<div class="flex-1 relative h-1.5 rounded-sm" style="background: rgba(215, 196, 131, 0.07);">
											<div
												class="absolute top-0 bottom-0 rounded-sm"
												style="width: {pct.toFixed(1)}%; opacity: 0.85; {isNeg ? 'right: 50%;' : 'left: 50%;'} max-width: 50%; background: {isNeg ? 'var(--rust-deep)' : 'var(--green-deep)'};"
											></div>
											<div
												class="absolute top-0 bottom-0 w-px"
												style="left: 50%; background: rgba(215, 196, 131, 0.18);"
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
								<Star size={11} fill="currentColor" style="color: var(--brand);" />
								closest matches from your ratings:
							</div>
							<div class="space-y-1">
								{#each explanation.top_contributing_rated_movies as [t, r] (t)}
									<div class="flex items-center gap-2">
										<Star size={10} fill="currentColor" style="color: var(--brand);" />
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
					class="btn btn-pill text-xs px-2.5 py-1 board"
					style="letter-spacing: 0.12em; {myVote === 'up'
						? 'color: var(--green); border: 1px solid rgba(143, 175, 122, 0.65); background: rgba(143, 175, 122, 0.07); text-shadow: 0 0 8px rgba(95, 135, 95, 0.6); box-shadow: 0 0 10px rgba(95, 135, 95, 0.2);'
						: 'background: var(--surface); border: 1px solid var(--border); color: var(--ink-muted);'}"
					aria-pressed={myVote === 'up'}
					onclick={() => clickVote('up')}
					title="Vote up as {voterName}"
				>
					<ThumbsUp size={11} />
					yes
				</button>
				<button
					type="button"
					class="btn btn-pill text-xs px-2.5 py-1 board"
					style="letter-spacing: 0.12em; {myVote === 'veto'
						? 'color: var(--rust); border: 1px solid rgba(187, 119, 68, 0.6); background: rgba(187, 119, 68, 0.07); text-shadow: 0 0 8px rgba(187, 119, 68, 0.55); box-shadow: 0 0 10px rgba(187, 119, 68, 0.18);'
						: 'background: var(--surface); border: 1px solid var(--border); color: var(--ink-muted);'}"
					aria-pressed={myVote === 'veto'}
					onclick={() => clickVote('veto')}
					title="Veto as {voterName}"
				>
					<Ban size={11} />
					veto
				</button>
				<span class="ml-1 text-[10px]" style="color: var(--ink-faint);">
					voting as <strong style="color: var(--ink-muted);">{voterName}</strong>
				</span>
			</div>
		{/if}
	</div>
</article>
