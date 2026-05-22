<script lang="ts">
	import { goto } from '$app/navigation';
	import ModeSelector from '$lib/components/ModeSelector.svelte';
	import StrategySelector from '$lib/components/StrategySelector.svelte';
	import RecommendationCard from '$lib/components/RecommendationCard.svelte';
	import {
		getModes,
		getStrategies,
		recommendGroup,
		type GroupRecResponse,
		type Mode,
		type Strategy
	} from '$lib/api';
	import { loadMembers } from '$lib/store';

	const members = $state(loadMembers());

	let modes = $state<Mode[]>([]);
	let strategies = $state<Strategy[]>([]);
	let mode = $state('balanced');
	let strategy = $state('group_taste_vector');
	let topN = $state(12);
	let excludeRated = $state(true);
	let excludeWatched = $state(true);

	let busy = $state(false);
	let error = $state<string | null>(null);
	let result = $state<GroupRecResponse | null>(null);

	$effect(() => {
		Promise.all([getModes(), getStrategies()]).then(([m, s]) => {
			modes = m.modes;
			strategies = s.strategies;
		});
	});

	async function run() {
		error = null;
		if (members.length < 2) {
			error = 'Add at least 2 group members on the landing page first.';
			return;
		}
		busy = true;
		try {
			result = await recommendGroup({
				hashes: members.map((m) => m.hash),
				member_names: members.map((m) => m.name),
				strategy,
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

	$effect(() => {
		if (members.length >= 2 && !result && !busy) run();
	});
</script>

<section class="space-y-6">
	<header class="flex items-center justify-between flex-wrap gap-3">
		<div>
			<h1 class="text-2xl font-semibold tracking-tight">Group picks</h1>
			<p class="text-sm text-white/50">
				{members.length} members · {members.map((m) => m.name).join(' + ') || '—'}
			</p>
		</div>
		<div class="flex gap-3 text-sm">
			<a href="/group/compatibility" class="text-sky-300 hover:text-sky-200"
				>compatibility report →</a
			>
			<button
				type="button"
				onclick={() => goto('/')}
				class="text-white/40 hover:text-white"
			>
				← add more
			</button>
		</div>
	</header>

	{#if members.length < 2}
		<div class="rounded-lg bg-red-500/10 border border-red-400/30 p-4 text-sm">
			You need at least 2 group members. <a href="/" class="underline">Go add some →</a>
		</div>
	{:else}
		<div class="rounded-lg bg-white/5 border border-white/10 p-4 space-y-4">
			<StrategySelector {strategies} bind:value={strategy} />
			<ModeSelector {modes} bind:value={mode} label="Re-rank mode (per member)" />
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
					<input type="checkbox" bind:checked={excludeRated} /> exclude rated
				</label>
				<label class="flex items-center gap-2 text-white/70">
					<input type="checkbox" bind:checked={excludeWatched} /> exclude watched
				</label>
				<button
					type="button"
					onclick={run}
					disabled={busy}
					class="ml-auto px-4 py-1.5 rounded bg-sky-500 text-sky-950 font-medium hover:bg-sky-400 disabled:opacity-50"
				>
					{busy ? 'computing…' : 'recompute'}
				</button>
			</div>
		</div>

		{#if error}
			<p class="text-red-400">{error}</p>
		{/if}

		{#if result}
			<div class="grid grid-cols-1 md:grid-cols-2 gap-3">
				{#each result.recommendations as r (r.movie_id)}
					<RecommendationCard
						title={r.title}
						score={r.score}
						explanation={r.explanation}
						per_member_score={r.per_member_score}
						fairness={r.fairness}
						memberOrder={members.map((m) => m.name)}
					/>
				{/each}
			</div>
		{/if}
	{/if}
</section>
