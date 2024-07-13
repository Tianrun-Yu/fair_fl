import torch
from torch import nn

from ....core.alg_frame.client_trainer import ClientTrainer


class MyModelTrainer(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate)
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                # logging.info("x.size = " + str(x.size()))
                # logging.info("labels.size = " + str(labels.size()))
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                # to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
                #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
            #     self.client_idx, epoch, sum(epoch_loss) / len(epoch_loss)))

    def test(self, test_data, device, args):
        print('我是test我运行了1')
        model = self.model
        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}
        criterion = nn.CrossEntropyLoss().to(device)

        target_list = []
        s_list = []
        x_list = []
        pred_list = []

        with torch.no_grad():
            for x, target, s in test_data:
                target_list.extend(target.tolist())
                s_list.extend(s.tolist())
                x_list.extend(x.tolist())

                x = x.to(device)
                target = target.to(device)
                s = s.to(device)
                print('我是test我运行了2')
                print(s)
                logits = model(x)
                loss = criterion(logits, target)

                _, predicted = torch.max(logits, -1)

                correct = predicted.eq(target).sum()
                pred_list.extend(predicted.detach().cpu().tolist())
                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)

        target_list = np.array(target_list)
        s_list = np.array(s_list)
        x_list = np.array(x_list)
        pred_list = np.array(pred_list)
        pred_acc = pred_list == target_list

        ppr_list = []
        tnr_list = []
        tpr_list = []
        converted_s = s_list[:, 1]  # sex, 1 attribute

        for s_value in np.unique(converted_s):
            if np.mean(converted_s == s_value) > 0.01:
                indexs0 = np.logical_and(target_list == 0, converted_s == s_value)
                indexs1 = np.logical_and(target_list == 1, converted_s == s_value)
                ppr_list.append(np.mean(pred_list[converted_s == s_value]))
                tnr_list.append(np.mean(pred_acc[indexs0]))
                tpr_list.append(np.mean(pred_acc[indexs1]))

        eo_gap = max(max(tnr_list) - min(tnr_list), max(tpr_list) - min(tpr_list))
        dp_gap = max(ppr_list) - min(ppr_list)

        metrics["eo_gap"] = eo_gap
        metrics["dp_gap"] = dp_gap

        return metrics
    """""
    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            "test_correct": 0,
            "test_loss": 0,
            "test_precision": 0,
            "test_recall": 0,
            "test_total": 0,
        }

    
        stackoverflow_lr is the task of multi-label classification
        please refer to following links for detailed explainations on cross-entropy and corresponding implementation of tff research:
        https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
        https://github.com/google-research/federated/blob/49a43456aa5eaee3e1749855eed89c0087983541/optimization/stackoverflow_lr/federated_stackoverflow_lr.py#L131
      
        if args.dataset == "stackoverflow_lr":
            criterion = nn.BCELoss(reduction="sum").to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                if args.dataset == "stackoverflow_lr":
                    predicted = (pred > 0.5).int()
                    correct = predicted.eq(target).sum(axis=-1).eq(target.size(1)).sum()
                    true_positive = ((target * predicted) > 0.1).int().sum(axis=-1)
                    precision = true_positive / (predicted.sum(axis=-1) + 1e-13)
                    recall = true_positive / (target.sum(axis=-1) + 1e-13)
                    metrics["test_precision"] += precision.sum().item()
                    metrics["test_recall"] += recall.sum().item()
                else:
                    _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                if len(target.size()) == 1:  #
                    metrics["test_total"] += target.size(0)
                elif len(target.size()) == 2:  # for tasks of next word prediction
                    metrics["test_total"] += target.size(0) * target.size(1)
        return metrics
    """""
    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        return False
