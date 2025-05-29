% Facts
parent(john, mary).
parent(mary, susan).
parent(john, mike).
parent(mike, kevin).
parent(lisa, john).
parent(james, lisa).
parent(mary, tom).
parent(mike, alice).
parent(alice, emma).
parent(tom, rachel).

% Rule
grandparent(X, Y) :- parent(X, Z), parent(Z, Y).
