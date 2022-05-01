/**
 * Framework for NoGo and similar games (C++ 11)
 * agent.h: Define the behavior of variants of the player
 *
 * Author: Theory of Computer Games (TCG 2021)
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include "board.h"
#include "action.h"
#include <fstream>
#include <time.h>

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * random player for both side
 * put a legal piece randomly
 */
class player : public random_agent {
public:
	player(const std::string& args = "") : random_agent("name=random role=unknown " + args),
		space(board::size_x * board::size_y), who(board::empty) {
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		if (role() == "black") who = board::black;
		if (role() == "white") who = board::white;
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
		//std::cout<< action::place(i,board::white) <<std::endl;
	}

	virtual action take_action(const board& state) {
		//std::cout<<"random state:"<<state<<std::endl;
		std::shuffle(space.begin(), space.end(), engine);
		for (const action::place& move : space) {
			board after = state;
			if (move.apply(after) == board::legal)
				return move;
		}
		return action();
	}

private:
	std::vector<action::place> space;
	board::piece_type who;
};

class Node
{
	public:
		board state;
		int win_count = 0;
		int visit_count = 0;
		double value = std::numeric_limits<float>::max();
		Node* parent = nullptr;
		action::place last_action;
		std::vector<Node*> children;
		board::piece_type node_who;

		~Node(){}
};

class MCTS_player : public random_agent {
public:
	MCTS_player(const std::string& args = "") : random_agent("name=random role=unknown " + args),
		white_space(board::size_x * board::size_y),
		black_space(board::size_x * board::size_y), 
		who(board::empty) {

		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		if (role() == "black") who = board::black;
		if (role() == "white") who = board::white;
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < white_space.size(); i++)
			white_space[i] = action::place(i, board::white);
		for (size_t i = 0; i < black_space.size(); i++)
			black_space[i] = action::place(i, board::black);
	}

	// value = win_count / visit_vount + 1.41 * UCB
	void compute_value(Node* node, int total_visit_count){
		node->value = ((double)node->win_count/node->visit_count) + 0.5 * sqrt(log((double)total_visit_count)/node->visit_count);
	}

	Node* select(Node* node){
		//std::cout<<"node_children size: "<<node->children.size()<<std::endl;
		while(node->children.empty() == false){
			double max_value = 0;
			int select_index = 0; 
			for(size_t i=0; i < node->children.size(); i++){
				if (max_value < node->children[i]->value){
					max_value = node->children[i]->value;
					select_index = i;
				}
			}
			node = node->children[select_index];
		}
		return node;
	}

	void expand(Node* parent_node){
		board::piece_type child_who;
		action::place child_move;
		child_who = (parent_node->node_who == board::black ? board::white : board::black);
		if (parent_node->node_who == board::black){
			child_who = board::white;
			for (const action::place& child_move : white_space){
				board after = parent_node->state;
				if (child_move.apply(after) == board::legal){
					Node* child_node = new Node;
					child_node->node_who = child_who;
					child_node->state = after;
					child_node->parent = parent_node;
					child_node->last_action = child_move;
					parent_node->children.push_back(child_node);
				}
			}
		}
		else if(parent_node->node_who == board::white){
			child_who = board::black;
			for (const action::place& child_move : black_space){
				board after = parent_node->state;
				if (child_move.apply(after) == board::legal){
					Node* child_node = new Node;
					child_node->node_who = child_who;
					child_node->state = after;
					child_node->parent = parent_node;
					child_node->last_action = child_move;
					parent_node->children.push_back(child_node);
				}
			}
		}
	}
	// simulation
	board::piece_type simulation(Node* node){
		bool finish = false;
		board state = node->state;
		//std::cout<<state<<std::endl;
		//int i ;
		//std::cin>>i;
		board::piece_type who = node->node_who;
		while(finish == false){
			//std::cout<<state<<std::endl;
			//std::cin>>i;
			who = (who == board::white ? board::black : board::white);
			finish = true;
			if (who == board::black){
				std::shuffle(black_space.begin(), black_space.end(), engine);
				for (const action::place& move : black_space) {
					board after = state;
					if (move.apply(after) == board::legal){
						move.apply(state);
						finish = false;
						break;
					}
				}
			}
			else if (who == board::white){
				std::shuffle(white_space.begin(), white_space.end(), engine);
				for (const action::place& move : white_space) {
					board after = state;
					if (move.apply(after) == board::legal){
						move.apply(state);
						finish = false;
						break;
					}
				}
			}
		}
		return (who == board::white ? board::black : board::white);
	}

	void backpropogation(Node* root, Node* node, board::piece_type winner, int total_visit_count){
		bool win = true;
		if (winner == root->node_who)
			win = false;
		while(node != nullptr){
			node->visit_count = node->visit_count + 1;
			if (win == true)
				node->win_count = node->win_count + 1;
			compute_value(node, total_visit_count);
			node = node->parent;
		}
	}

	void delete_tree(Node* node){
		if(node->children.empty() == false){
			for(size_t i = 0; i < node->children.size(); i++){
				delete_tree(node->children[i]);
				if (node->children[i] != nullptr)
					free(node->children[i]);	
			}
			node->children.clear();
		}
	}

	action greedy_select(Node* node){
		int child_index = -1;
		int max_visit_count = 0;
		for(size_t i = 0; i < node->children.size(); i++){
			if (node->children[i]->visit_count > max_visit_count){
				max_visit_count = node->children[i]->visit_count;
				child_index = i;
			}
		}
		if (child_index == -1)
			return action();
		else
			return node->children[child_index]->last_action;
	}
	
	virtual action take_action(const board& state){
		clock_t start_time, end_time;
		start_time = clock();
		Node* root = new Node;
		board::piece_type winner;
		double total_time = 0;
		int total_visit_count = 0;
		int remain_empty = 0;
		for(int i = 0; i < 9; i++){
			for(int j = 0; j < 9; j++){
				if(state[i][j] == board::empty)
					remain_empty++;
			}
		}
		step_count = 36 - remain_empty / 2;

		root->state = state;
		root->node_who = (who == board::white ? board::black : board::white);
		expand(root);
		while(total_time < 0.95 * time_schedule[step_count]){
			Node* greedy_node;
			greedy_node = select(root);
			expand(greedy_node);
			winner = simulation(greedy_node);
			//std::cout<<winner<<std::endl;
			total_visit_count = total_visit_count + 1;
			backpropogation(root, greedy_node, winner, total_visit_count);
			end_time = clock();
			total_time = (double)(end_time - start_time)/CLOCKS_PER_SEC;
		}
		action result = greedy_select(root);
		delete_tree(root);
		free(root);
		return result;
	}
private:
	double time_schedule[36] = {0.2, 0.2, 0.2, 0.4, 0.4, 0.4,
								0.7, 0.7, 0.7, 1.4, 1.4, 1.4,
						 		1.7, 1.7, 1.7, 2.0, 2.0, 2.0,
						 		1.7, 1.7, 1.7, 1.7, 1.7, 1.7,
						 		1.0, 1.0, 1.0, 0.5, 0.5, 0.5,
						 		0.4, 0.4, 0.4, 0.2, 0.2, 0.2 };
	int step_count = 0;
	std::vector<action::place> white_space;
	std::vector<action::place> black_space;
	board::piece_type who;
};


