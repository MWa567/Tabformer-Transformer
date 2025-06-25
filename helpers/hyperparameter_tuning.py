# Hyperparameter tuning
import optuna

# Define objective, then define trial where parameters can vary, then study.optimize

def get_best_params(train_data, val_data):
  trial_data = []

  def objective(trial):
    # Define parameters to optimize
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True),
        'd_model': trial.suggest_int('d_model', 16, 128),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'num_layers': trial.suggest_int('num_layers', 1, 10),
    }

    # Train model
    np.random.seed(42)

    transformer = Transformer(categorical, numerical, params['num_layers'], max_len+1, params['d_model'] * 4, 1, 4, 128, params['dropout']) # parameters in the custom format for my transformer class
    criterion_cat = nn.CrossEntropyLoss()
    criterion_num = nn.MSELoss()
    optimizer = optim.Adam(transformer.parameters(), lr=params['learning_rate'])

    transformer.train()

    total_loss = 0
    for epoch in range(3):
      for user in range(len(train_data) // 2):
        optimizer.zero_grad()

        train_input = train_tgt_data[user, :-1, :]  # (seq_len - 1, num_features)

        train_target = train_tgt_data[user, 1:, :]  # (seq_len - 1, num_features)

        _, preds_cat, preds_num = transformer(train_input)

        train_tgt_cat = train_target[:, :len(categorical)].long()  # (seq_len - 1, num_categorical)
        train_tgt_num = train_target[:, len(categorical):len(categorical) + numerical].float()  # (seq_len - 1, num_numeric)

        # Loss calculations
        cat_loss = sum(
            criterion_cat(pred.squeeze(0)[: train_num_trans_per_user[user]-1], train_tgt_cat[:, i][1: train_num_trans_per_user[user]].to(pred.device))
            for i, pred in enumerate(preds_cat)
        ) / len(preds_cat)

        num_loss = sum(
            criterion_num(pred.squeeze(0).squeeze(1)[: train_num_trans_per_user[user]-1], train_tgt_num[:, i][1: train_num_trans_per_user[user]])
            for i, pred in enumerate(preds_num)
        ) / len(preds_num)

        loss = cat_loss * 0.85 + num_loss * 0.15
        loss.backward()
        optimizer.step()

    # Validation set
    transformer.eval()
    total_val_loss = 0
    with torch.no_grad():
      for epoch in range(3):
        for user in range(len(val_data) // 2):
          val_input = val_tgt_data[user, :-1, :]
          val_target = val_tgt_data[user, 1:, :]

          _, preds_cat, preds_num = transformer(val_input)

          tgt_cat = val_target[:, :len(categorical)].long()
          tgt_num = val_target[:, len(categorical):len(categorical) + numerical].float()

          cat_loss = sum(
              criterion_cat(pred.squeeze(0)[: val_num_trans_per_user[user]-1],
                            tgt_cat[:, i][1: val_num_trans_per_user[user]].to(pred.device))
              for i, pred in enumerate(preds_cat)
          ) / len(preds_cat)

          num_loss = sum(
              criterion_num(pred.squeeze(0).squeeze(1)[: val_num_trans_per_user[user]-1],
                            tgt_num[:, i][1: val_num_trans_per_user[user]])
              for i, pred in enumerate(preds_num)
          ) / len(preds_num)

          val_loss = cat_loss * 0.85 + num_loss * 0.15
          total_val_loss += val_loss.item()
    avg_val_loss = total_val_loss / len(val_data)
    return avg_val_loss

  study = optuna.create_study(direction='minimize')
  study.optimize(objective, n_trials=25, show_progress_bar=True)

  return study.best_params
