<script lang="ts">
	import { page } from '$app/state';
	import { fly } from 'svelte/transition';
	import { Loader2, ArrowRight, Heart, AlertTriangle } from 'lucide-svelte';
	import Avatar from '$lib/components/Avatar.svelte';
	import Stat from '$lib/components/Stat.svelte';
	import { analyzeGroup, getGroup, type GroupAnalysis, type GroupState } from '$lib/api';
	import { loadMembers } from '$lib/store';

	// Same dual-mode as /group/recommendations: ?group=ID hydrates from the
	// server, otherwise fall back to localStorage members.
	const groupId = $derived(page.url.searchParams.get('group'));
	let serverGroup = $state<GroupState | null>(null);
	const localMembers = $state(loadMembers());

	const members = $derived(
		serverGroup
			? serverGroup.members.map((m) => ({ name: m.name, hash: m.hash }))
			: localMembers.map((m) => ({ name: m.name, hash: m.hash }))
	);

	let busy = $state(true);
	let error = $state<string | null>(null);
	let report = $state<GroupAnalysis | null>(null);

	$effect(() => {
		if (groupId) {
			getGroup(groupId)
				.then((g) => (serverGroup = g))
				.catch((e) => {
					error = e instanceof Error ? e.message : String(e);
					busy = false;
				});
		}
	});

	$effect(() => {
		// Wait for the server group to land if one was requested.
		if (groupId && !serverGroup) return;
		if (members.length < 2) {
			busy = false;
			return;
		}
		busy = true;
		report = null;
		analyzeGroup({
			hashes: members.map((m) => m.hash),
			member_names: members.map((m) => m.name),
			top_overlap: 10
		})
			.then((r) => (report = r))
			.catch((e) => (error = e instanceof Error ? e.message : String(e)))
			.finally(() => (busy = false));
	});

	function colorFor(v: number | null): string {
		if (v === null) return 'var(--ink-dim)';
		if (v >= 0.5) return '#34d399';
		if (v >= 0.3) return '#86efac';
		if (v >= 0) return 'var(--ink-muted)';
		if (v >= -0.3) return '#fdba74';
		return '#f87171';
	}

	function labelFor(v: number | null): string {
		if (v === null) return 'n/a';
		if (v >= 0.6) return 'aligned';
		if (v >= 0.3) return 'similar';
		if (v >= 0) return 'mild';
		if (v >= -0.3) return 'mixed';
		return 'opposite';
	}

	function pairwiseHighlights(report: GroupAnalysis) {
		const ranked = [...report.pairwise].sort(
			(a, b) => (b.pearson_on_shared ?? -2) - (a.pearson_on_shared ?? -2)
		);
		return {
			best: ranked[0],
			worst: ranked[ranked.length - 1]
		};
	}
</script>

<svelte:head>
	<title>Compatibility · movienight</title>
</svelte:head>

<section class="space-y-6 anim-fade-up">
	<header class="flex items-end justify-between flex-wrap gap-3">
		<div>
			<span class="chip chip-violet">Group · compatibility report</span>
			<h1 class="display-md mt-2" style="font-family: 'Playfair Display', Georgia, serif; font-style: italic;">
				How <span class="text-gradient">aligned</span> are you?
			</h1>
			<p class="text-sm" style="color: var(--ink-muted);">
				Pearson correlation on films you've all seen + cosine similarity of your TF-IDF taste
				vectors across the whole catalog.
			</p>
		</div>
		<a
			href={groupId ? `/group/recommendations?group=${encodeURIComponent(groupId)}` : '/group/recommendations'}
			class="btn btn-ghost btn-pill text-xs"
		>
			back to recs
			<ArrowRight size={13} />
		</a>
	</header>

	{#if groupId && !serverGroup}
		<div class="surface p-5 text-sm" style="color: var(--ink-muted);">
			<Loader2 size={14} class="animate-spin inline mr-2" />
			Loading shared group <code>{groupId}</code>…
		</div>
	{:else if members.length < 2}
		<div class="surface p-5 text-sm" style="background: var(--rose-dim); border-color: rgba(248,113,113,0.3);">
			Need at least 2 members.
			{#if groupId}
				Open <a href="/group/{groupId}/join" class="underline" style="color: #fca5a5;">the join link →</a>
			{:else}
				<a href="/" class="underline" style="color: #fca5a5;">Add some →</a>
			{/if}
		</div>
	{:else if busy}
		<div class="surface p-8 text-center">
			<Loader2 size={20} class="animate-spin mx-auto mb-2" style="color: var(--violet);" />
			<p class="text-sm" style="color: var(--ink-muted);">computing pairwise overlaps…</p>
		</div>
	{:else if error}
		<p class="text-sm rounded-md px-3 py-2" style="background: var(--rose-dim); border: 1px solid rgba(248,113,113,0.3); color: #fecaca;">{error}</p>
	{:else if report}
		{@const highlights = pairwiseHighlights(report)}

		{#if highlights.best || highlights.worst}
			<div class="grid grid-cols-1 md:grid-cols-2 gap-3 anim-fade-up">
				{#if highlights.best}
					<div class="surface p-4 anim-fade-up flex items-center gap-3">
						<div class="grid place-items-center w-10 h-10 rounded-full" style="background: rgba(52,211,153,0.18);">
							<Heart size={18} style="color: #6ee7b7;" />
						</div>
						<div>
							<div class="text-[10px] uppercase tracking-[0.14em]" style="color: var(--ink-faint);">
								most aligned
							</div>
							<div class="text-base font-medium">
								{highlights.best.pair[0]} ↔ {highlights.best.pair[1]}
							</div>
							<div class="text-xs mono" style="color: #6ee7b7;">
								pearson {highlights.best.pearson_on_shared?.toFixed(2) ?? 'n/a'} ·
								cosine {highlights.best.cosine_on_taste?.toFixed(2) ?? 'n/a'} ·
								{highlights.best.n_shared_movies} shared films
							</div>
						</div>
					</div>
				{/if}
				{#if highlights.worst && highlights.worst !== highlights.best}
					<div class="surface p-4 anim-fade-up flex items-center gap-3" style="animation-delay: 60ms;">
						<div class="grid place-items-center w-10 h-10 rounded-full" style="background: rgba(248,113,113,0.15);">
							<AlertTriangle size={18} style="color: #fca5a5;" />
						</div>
						<div>
							<div class="text-[10px] uppercase tracking-[0.14em]" style="color: var(--ink-faint);">
								biggest tension
							</div>
							<div class="text-base font-medium">
								{highlights.worst.pair[0]} ↔ {highlights.worst.pair[1]}
							</div>
							<div class="text-xs mono" style="color: #fca5a5;">
								pearson {highlights.worst.pearson_on_shared?.toFixed(2) ?? 'n/a'} ·
								cosine {highlights.worst.cosine_on_taste?.toFixed(2) ?? 'n/a'} ·
								{highlights.worst.n_shared_movies} shared films
							</div>
						</div>
					</div>
				{/if}
			</div>
		{/if}

		<div class="surface p-5 anim-fade-up">
			<h2 class="text-lg font-medium mb-3 flex items-center gap-2">Pairwise overlap</h2>
			<div class="space-y-2">
				{#each report.pairwise as p (p.pair.join('+'))}
					<div
						class="flex items-center gap-3 px-3 py-2 rounded-lg"
						style="background: rgba(0,0,0,0.2); border: 1px solid var(--border);"
					>
						<div class="flex items-center -space-x-2">
							<Avatar name={p.pair[0]} size={28} />
							<Avatar name={p.pair[1]} size={28} />
						</div>
						<div class="flex-1 min-w-0">
							<div class="text-sm">
								<strong>{p.pair[0]}</strong>
								<span style="color: var(--ink-dim);">↔</span>
								<strong>{p.pair[1]}</strong>
							</div>
							<div class="text-[11px] mono" style="color: var(--ink-faint);">
								{p.n_shared_movies} shared films
							</div>
						</div>
						<div class="flex gap-4 items-center">
							<div class="text-right">
								<div class="text-[10px] uppercase tracking-[0.12em]" style="color: var(--ink-faint);">
									pearson
								</div>
								<div class="font-medium tabular-nums" style="color: {colorFor(p.pearson_on_shared)};">
									{p.pearson_on_shared?.toFixed(2) ?? '—'}
								</div>
							</div>
							<div class="text-right">
								<div class="text-[10px] uppercase tracking-[0.12em]" style="color: var(--ink-faint);">
									cosine
								</div>
								<div class="font-medium tabular-nums" style="color: {colorFor(p.cosine_on_taste)};">
									{p.cosine_on_taste?.toFixed(2) ?? '—'}
								</div>
							</div>
							<span
								class="chip"
								style="background: {colorFor(p.pearson_on_shared)}33; color: {colorFor(p.pearson_on_shared)}; border-color: {colorFor(p.pearson_on_shared)}55;"
							>
								{labelFor(p.pearson_on_shared)}
							</span>
						</div>
					</div>
				{/each}
			</div>
			<p class="text-xs mt-3" style="color: var(--ink-faint);">
				Pearson = correlation on films both have rated. Cosine = similarity of TF-IDF taste vectors.
			</p>
		</div>

		<div class="grid grid-cols-1 md:grid-cols-2 gap-4 anim-fade-up" style="animation-delay: 100ms;">
			<div class="surface p-5">
				<div class="flex items-center gap-2 mb-3">
					<Heart size={16} style="color: var(--brand);" />
					<h2 class="text-base font-medium">Consensus picks</h2>
				</div>
				<p class="text-xs mb-3" style="color: var(--ink-faint);">
					Films you all rated ≥ 3.5★ with tight agreement (σ ≤ 0.75).
				</p>
				{#if report.consensus_movies.length === 0}
					<p class="text-xs italic px-2 py-3" style="color: var(--ink-faint);">
						No clear consensus yet — either too few shared films, or no film cleared the bar
						(mean ≥ 3.5★, σ ≤ 0.75). Try after everyone's added more ratings.
					</p>
				{:else}
				<ul class="space-y-1.5 text-sm">
					{#each report.consensus_movies as m, i (m.movie_id)}
						<li
							class="flex items-baseline gap-3 py-1.5 px-2 rounded-md"
							style="background: rgba(0,0,0,0.15);"
							in:fly={{ y: 4, delay: i * 30, duration: 280 }}
						>
							<span class="text-[11px] tabular-nums mono" style="color: var(--ink-faint);">
								#{i + 1}
							</span>
							<div class="flex-1 min-w-0">
								<div class="font-medium truncate">{m.title}</div>
								<div class="text-[10px] mono" style="color: var(--ink-dim);">
									{Object.entries(m.member_ratings).map(([n, r]) => `${n} ${r}★`).join(' · ')}
								</div>
							</div>
							<span class="mono text-xs" style="color: #6ee7b7;">
								{m.mean.toFixed(1)}★
							</span>
						</li>
					{/each}
				</ul>
				{/if}
			</div>

			<div class="surface p-5">
				<div class="flex items-center gap-2 mb-3">
					<AlertTriangle size={16} style="color: var(--rose);" />
					<h2 class="text-base font-medium">Disagreement picks</h2>
				</div>
				<p class="text-xs mb-3" style="color: var(--ink-faint);">
					Films where one member loves it and another would skip (σ ≥ 1.0★).
				</p>
				{#if report.disagreement_movies.length === 0}
					<p class="text-xs italic px-2 py-3" style="color: var(--ink-faint);">
						You're too aligned for real disagreement — no shared film has σ ≥ 1.0★.
						{#if report.consensus_movies.length > 0}
							That's a good sign for movie night.
						{/if}
					</p>
				{:else}
				<ul class="space-y-1.5 text-sm">
					{#each report.disagreement_movies as m, i (m.movie_id)}
						<li
							class="flex items-baseline gap-3 py-1.5 px-2 rounded-md"
							style="background: rgba(0,0,0,0.15);"
							in:fly={{ y: 4, delay: i * 30, duration: 280 }}
						>
							<span class="text-[11px] tabular-nums mono" style="color: var(--ink-faint);">
								#{i + 1}
							</span>
							<div class="flex-1 min-w-0">
								<div class="font-medium truncate">{m.title}</div>
								<div class="text-[10px] mono" style="color: var(--ink-dim);">
									{Object.entries(m.member_ratings).map(([n, r]) => `${n} ${r}★`).join(' · ')}
								</div>
							</div>
							<span class="mono text-xs" style="color: #fca5a5;">
								σ {m.std.toFixed(1)}
							</span>
						</li>
					{/each}
				</ul>
				{/if}
			</div>
		</div>
	{/if}
</section>
