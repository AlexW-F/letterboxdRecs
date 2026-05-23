<script lang="ts">
	import { goto } from '$app/navigation';
	import { fly } from 'svelte/transition';
	import {
		Sparkles,
		Loader2,
		RefreshCw,
		ArrowLeft,
		BarChart3,
		Users as UsersIcon
	} from 'lucide-svelte';
	import ModeSelector from '$lib/components/ModeSelector.svelte';
	import StrategySelector from '$lib/components/StrategySelector.svelte';
	import RecommendationCard from '$lib/components/RecommendationCard.svelte';
	import Avatar from '$lib/components/Avatar.svelte';
	import Stat from '$lib/components/Stat.svelte';
	import Shimmer from '$lib/components/Shimmer.svelte';
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

	const totalRated = $derived(
		members.reduce((acc, m) => acc + (m.upload?.n_ratings_mapped ?? 0), 0)
	);
</script>

<section class="space-y-6 anim-fade-up">
	<header class="flex items-end justify-between flex-wrap gap-3">
		<div>
			<span class="chip chip-violet">
				<UsersIcon size={11} />
				Group · {members.length} members · {strategy}
			</span>
			<h1 class="display-md mt-2" style="font-family: 'Instrument Serif', Georgia, serif; font-style: italic;">
				Picks <span class="text-gradient">your group</span> will love.
			</h1>
			<div class="mt-2 flex flex-wrap items-center gap-2">
				{#each members as m (m.hash)}
					<div class="flex items-center gap-1.5 chip" style="padding: 0.2rem 0.5rem 0.2rem 0.2rem;">
						<Avatar name={m.name} size={20} />
						<span class="text-xs normal-case" style="color: var(--ink-muted); letter-spacing: 0;">
							{m.name}
						</span>
					</div>
				{/each}
			</div>
		</div>
		<div class="flex gap-2 text-sm">
			<a href="/group/compatibility" class="btn btn-ghost btn-pill text-xs">
				<BarChart3 size={13} />
				compatibility
			</a>
			<button class="btn btn-ghost btn-pill text-xs" onclick={() => goto('/')}>
				<ArrowLeft size={13} />
				add more
			</button>
		</div>
	</header>

	{#if members.length < 2}
		<div class="surface p-5 text-sm" style="background: var(--rose-dim); border-color: rgba(248,113,113,0.3);">
			You need at least 2 group members. <a href="/" class="underline" style="color: #fca5a5;">Go add some →</a>
		</div>
	{:else}
		<div class="grid grid-cols-2 md:grid-cols-4 gap-3 anim-fade-up">
			<Stat label="group size" value={members.length} hint="uploaded friends" />
			<Stat label="total ratings" value={totalRated.toLocaleString()} hint="across the group" />
			<Stat label="strategy" value={strategy.replace(/_/g, ' ')} hint="how votes combine" />
			<Stat label="mode" value={mode} hint="re-rank tone per member" />
		</div>

		<div class="surface p-5 space-y-4 anim-fade-up" style="animation-delay: 60ms;">
			<StrategySelector {strategies} bind:value={strategy} />
			<ModeSelector {modes} bind:value={mode} label="Re-rank mode (applied per member)" />
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
				<button class="ml-auto btn btn-secondary" onclick={run} disabled={busy}>
					{#if busy}
						<Loader2 size={14} class="animate-spin" />
						computing
					{:else}
						<Sparkles size={14} />
						recompute
					{/if}
				</button>
			</div>
		</div>

		{#if error}
			<p class="text-sm rounded-md px-3 py-2" style="background: var(--rose-dim); border: 1px solid rgba(248,113,113,0.3); color: #fecaca;">{error}</p>
		{/if}

		{#if busy && !result}
			<div class="grid grid-cols-1 md:grid-cols-2 gap-3 anim-fade-up">
				{#each Array(8) as _, i (i)}
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
						per_member_score={r.per_member_score}
						fairness={r.fairness}
						memberOrder={members.map((m) => m.name)}
					/>
				{/each}
			</div>
		{/if}
	{/if}
</section>
