import test from 'node:test';
import assert from 'node:assert/strict';
import {
  generateSyntheticNetwork,
  extractAccountFeaturesAtCutoff,
  runFullSimulation,
  computeAccountShapExplanation,
  computeGnnExplainerSubgraph,
} from './fraudEngine';

test('generator determinism: identical seeds produce identical networks and rings', () => {
  const netA = generateSyntheticNetwork({ seed: 42, fraudRate: 0.015 });
  const netB = generateSyntheticNetwork({ seed: 42, fraudRate: 0.015 });
  const netC = generateSyntheticNetwork({ seed: 99, fraudRate: 0.015 });

  assert.deepEqual(netA.accounts, netB.accounts);
  assert.deepEqual(netA.rings, netB.rings);
  assert.equal(netA.transactions.length, netB.transactions.length);
  assert.equal(netA.transactions[0].amount, netB.transactions[0].amount);
  assert.notEqual(netA.transactions[0].timestamp, netC.transactions[0].timestamp);
});

test('all four fraud ring patterns are planted across the network', () => {
  const net = generateSyntheticNetwork({ seed: 42, fraudRate: 0.015 });
  const patterns = new Set(net.rings.map((r) => r.pattern));
  assert.equal(patterns.has('money_mule_chain'), true);
  assert.equal(patterns.has('shared_device_ring'), true);
  assert.equal(patterns.has('burst_activity'), true);
  assert.equal(patterns.has('circular_transfer'), true);
});

test('strict temporal split & zero future data leakage check passes', () => {
  const bundle = runFullSimulation({ seed: 42, fraudRate: 0.015, shiftedTactics: false });
  assert.equal(bundle.leakageReport.passed, true);
  assert.equal(bundle.leakageReport.strictlyOrderedTimestamps, true);
  assert.equal(bundle.leakageReport.zeroFutureFeatureLeakage, true);
  assert.equal(
    bundle.leakageReport.trainMaxTimestamp < bundle.leakageReport.valMinTimestamp,
    true
  );
  assert.equal(
    bundle.leakageReport.valMaxTimestamp < bundle.leakageReport.testMinTimestamp,
    true
  );

  // Double-check feature invariance when future transactions are tampered with
  const cutoff = bundle.leakageReport.trainMaxTimestamp;
  const f1 = extractAccountFeaturesAtCutoff(
    bundle.accounts,
    bundle.devices,
    bundle.merchants,
    bundle.transactions,
    cutoff
  );
  const mutatedTx = bundle.transactions.map((tx) =>
    tx.timestamp > cutoff ? { ...tx, amount: 9999999 } : tx
  );
  const f2 = extractAccountFeaturesAtCutoff(
    bundle.accounts,
    bundle.devices,
    bundle.merchants,
    mutatedTx,
    cutoff
  );
  assert.deepEqual(f1, f2);
});

test('GNN models outperform tabular-only baseline on topological fraud patterns and produce valid explanations', () => {
  const bundle = runFullSimulation({ seed: 42, fraudRate: 0.015, shiftedTactics: false });
  const tabEval = bundle.standardEvaluation.find((m) => m.modelName.includes('Tabular Only'))!;
  const gatEval = bundle.standardEvaluation.find((m) => m.modelName.includes('GAT'))!;

  assert.equal(gatEval.prAuc > tabEval.prAuc, true);
  assert.equal(gatEval.recallAtP80 >= tabEval.recallAtP80, true);

  const fraudAcc = bundle.testFeatures.find((f) => f.isFraud)!;
  const shap = computeAccountShapExplanation(fraudAcc, bundle.testFeatures);
  assert.equal(shap.length > 0, true);

  const gnnSubgraph = computeGnnExplainerSubgraph(
    fraudAcc.accountId,
    bundle.accounts,
    bundle.transactions
  );
  assert.equal(gnnSubgraph.length > 0, true);
});
